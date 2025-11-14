from transformers import pipeline, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-large"
pipe = pipeline("text-generation", model=MODEL_NAME)
# Tokenizer usado para contar/truncar tokens do prompt
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def analyze_cars(cars, max_prompt_tokens: int = 600):
    """Gera um prompt com instruções + lista de carros, garantindo que o prompt
    tenha no máximo `max_prompt_tokens` tokens antes de enviar ao pipeline.
    O limite padrão é 800 para deixar espaço para a resposta dentro do contexto do GPT-2.
    """
    base_prompt = """
### ANÁLISE DE CARROS DISPONÍVEIS

Responda SOMENTE em Português. Use APENAS os dados da LISTA DE CARROS DISPONÍVEIS abaixo (não busque informações externas).

Escolha EXPLICITAMENTE UM carro da lista como MELHOR RECOMENDAÇÃO.

RETORNE APENAS UM OBJETO JSON válido (sem texto adicional) com a seguinte estrutura:

{
  "melhor": { "modelo": "string", "preco": number, "site": "string", "local": "string", "motivo": "string", "pontos_fortes": ["string"], "cuidados": ["string"] },
  "alternativas": [ { "modelo": "string", "preco": number, "site": "string", "local": "string", "comparacao": "string" } ]
}

O campo "motivo" deve ter 1-2 frases conectando a escolha ao preço, km e localização quando possível.
Não inclua URLs, listas genéricas sobre o mercado brasileiro ou qualquer texto extra fora do JSON.

LISTA DE CARROS DISPONÍVEIS:
"""

    cars_list = "\n".join(
        f"- {c.get('modelo','')} ({c.get('site','')}) - R${c.get('preco','')} - Local: {c.get('local','')}" for c in cars
    )

    full_prompt = base_prompt + cars_list + "\n\nRECOMENDAÇÃO (use o formato acima):\n"

    # GPT-2 tem limite fixo de 1024 tokens
    model_max = 1024
    max_new_tokens = 200  # Reduzido para garantir que caiba no contexto

    # Garantir que temos espaço para o prompt + resposta
    allowed_input_tokens = min(model_max - max_new_tokens, max_prompt_tokens)
    if allowed_input_tokens <= 0:
        raise ValueError(f"Limite de tokens muito pequeno. Precisa de espaço para prompt e resposta.")

    # respeitar também o limite solicitado pelo usuário (max_prompt_tokens)
    allowed_input_tokens = min(allowed_input_tokens, max_prompt_tokens)

    # Tokenizar e truncar o prompt para `allowed_input_tokens` (preservando o começo)
    token_ids = tokenizer.encode(full_prompt)
    if len(token_ids) > allowed_input_tokens:
        token_ids = token_ids[:allowed_input_tokens]
    prompt_truncated = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    output = pipe(
        prompt_truncated,
        max_new_tokens=300,        # Ajustado para o tamanho da resposta esperada
        temperature=0.3,           # Reduzido para respostas mais previsíveis
        top_p=0.8,                # Mais restritivo para manter foco
        do_sample=True,
        truncation=True,
        no_repeat_ngram_size=3,    # Evita repetição de frases
        pad_token_id=50256,
        return_full_text=False     # Retorna apenas o texto gerado
    )

    gen = output[0]["generated_text"]

    # Tentar extrair JSON da resposta
    import re, json

    # procura primeiro objeto JSON na resposta
    m = re.search(r"\{.*\}", gen, re.S)
    if m:
        json_text = m.group(0)
        try:
            data = json.loads(json_text)
            # Formatar saída legível em Português a partir do JSON
            melhor = data.get("melhor", {})
            alt = data.get("alternativas", [])
            parts = []
            parts.append("=== RECOMENDAÇÃO DA IA ===")
            parts.append(f"MELHOR RECOMENDAÇÃO: {melhor.get('modelo','')} - R${melhor.get('preco','')} - {melhor.get('site','')}")
            parts.append("")
            parts.append(f"MOTIVO DA ESCOLHA: {melhor.get('motivo','')}")
            parts.append("")
            parts.append("PONTOS FORTES:")
            for p in melhor.get('pontos_fortes', []):
                parts.append(f"- {p}")
            parts.append("")
            parts.append("CUIDADOS NA COMPRA:")
            for c in melhor.get('cuidados', []):
                parts.append(f"- {c}")
            parts.append("")
            if alt:
                parts.append("OUTRAS OPÇÕES:")
                for a in alt[:2]:
                    parts.append(f"- {a.get('modelo','')} - R${a.get('preco','')} ({a.get('site','')}) - {a.get('comparacao','')}")

            return "\n".join(parts)
        except Exception:
            # parsing falhou; cair para fallback
            pass

    # Fallback determinístico aprimorado: pontuar carros por preço, consumo estimado e custo de manutenção
    try:
        # Função auxiliar: estima consumo (km/l) e manutenção anual (R$) a partir do nome/modelo
        def estimate_consumption_and_maintenance(modelo: str):
            m = (modelo or "").upper()
            # tentativa de extrair deslocamento do motor (ex: '1.0', '2.0', '3.2')
            import re
            disp_match = re.search(r"(\d+\.\d+)", m)
            disp = float(disp_match.group(1)) if disp_match else None

            # base km/l por deslocamento (valores heurísticos)
            if disp is None:
                km_l = 10.0
            elif disp <= 1.0:
                km_l = 14.0
            elif disp <= 1.3:
                km_l = 13.0
            elif disp <= 1.6:
                km_l = 12.0
            elif disp <= 2.0:
                km_l = 10.0
            else:
                km_l = 8.0

            # ajustes por tipo/keywords
            if any(x in m for x in ("DIESEL", "TURBO DIESEL")):
                km_l *= 1.25
            if any(x in m for x in ("TURBO", "TGDI", "TFSI", "TSI")):
                km_l *= 0.95
            if any(x in m for x in ("RANGER", "AMAROK", "S10", "RANGE", "PAJERO", "TROLLER", "T-CROSS", "TAOS", "TORO", "TUCSON")):
                # SUVs/pickups tendem a consumir mais
                km_l *= 0.75

            # estimativa de manutenção anual por segmento (heurística)
            if any(x in m for x in ("BMW", "VOLVO", "PORSCHE", "AUDI", "LAND ROVER", "MERCEDES")):
                manut = 8000
            elif any(x in m for x in ("JEEP", "TOYOTA", "HONDA", "NISSAN", "KIA", "HYUNDAI")):
                manut = 4000
            elif any(x in m for x in ("FIAT", "CHEVROLET", "RENAULT", "VOLKSWAGEN")):
                manut = 3000
            else:
                manut = 3500

            return max(1.0, km_l), manut

        # parâmetros para cálculo de custo total de uso
        ANNUAL_KM = 15000
        FUEL_PRICE = 6.0  # R$ por litro (heurístico)

        # calcular métricas para cada carro
        scored = []
        precios = [c.get('preco', 0) or 0 for c in cars]
        max_price = max(precios) if precios else 1

        fuel_costs = []
        manuts = []
        for c in cars:
            modelo = c.get('modelo') or c.get('veiculo') or ''
            km_l, manut = estimate_consumption_and_maintenance(modelo)
            # calcular custo anual de combustível
            fuel_cost = (ANNUAL_KM / km_l) * FUEL_PRICE
            fuel_costs.append(fuel_cost)
            manuts.append(manut)

        max_fuel = max(fuel_costs) if fuel_costs else 1
        max_manut = max(manuts) if manuts else 1

        for idx, c in enumerate(cars):
            preco = c.get('preco') or 0
            modelo = c.get('modelo') or c.get('veiculo') or ''
            km_l, manut = estimate_consumption_and_maintenance(modelo)
            fuel_cost = fuel_costs[idx]

            # normalizar (0..1)
            price_norm = preco / max_price
            fuel_norm = fuel_cost / max_fuel
            manut_norm = manut / max_manut

            # pesos — priorizar qualidade de uso sobre preço absoluto
            w_price = 0.45
            w_fuel = 0.35
            w_manut = 0.20

            score = price_norm * w_price + fuel_norm * w_fuel + manut_norm * w_manut
            scored.append((score, c, km_l, fuel_cost, manut))

        # escolher menor score
        scored.sort(key=lambda x: x[0])
        best_score, best_car, best_km_l, best_fuel_cost, best_manut = scored[0]

        parts = []
        parts.append(f"MELHOR RECOMENDAÇÃO: {best_car.get('modelo', best_car.get('veiculo',''))} - R${best_car.get('preco','')} - {best_car.get('site','')}")
        parts.append("")
        parts.append("MOTIVO DA ESCOLHA: Selecionado por um balanço entre preço, consumo estimado e custo de manutenção. A escolha prioriza menor custo total de uso anual considerando consumo de combustível e manutenção esperada.")
        parts.append("")
        parts.append("DETALHES ESTIMADOS:")
        parts.append(f"- Consumo estimado: {best_km_l:.1f} km/l")
        parts.append(f"- Custo anual de combustível estimado: R${best_fuel_cost:.0f}")
        parts.append(f"- Custo anual de manutenção estimado: R${best_manut}")
        parts.append("")
        parts.append("PONTOS FORTES:")
        parts.append(f"- Boa relação entre preço e custo de uso")
        parts.append("- Disponibilidade local conforme a listagem")
        parts.append("")
        parts.append("CUIDADOS NA COMPRA:")
        parts.append("- Confirmar consumo real no test-drive e histórico de revisões")
        parts.append("- Negociar verificação pré-compra (mecânico de confiança)")

        # inserir 1-2 alternativas próximas
        if len(scored) > 1:
            parts.append("")
            parts.append("ALTERNATIVAS RELEVANTES:")
            for _, alt, alt_km_l, alt_fuel_cost, alt_manut in scored[1:3]:
                parts.append(f"- {alt.get('modelo', alt.get('veiculo',''))} - R${alt.get('preco','')} | Consumo estimado: {alt_km_l:.1f} km/l | Manutenção anual: R${alt_manut}")

        return "\n".join(parts)
    except Exception:
        # último recurso: retornar o texto gerado bruto
        return gen