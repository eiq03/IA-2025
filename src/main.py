from api           import get_user_input
from scrapper      import get_cars_from_sources
from prompt_engine import analyze_cars
from formatter     import format_output

import re


def parse_valor_interval(raw: str):
    # retorna (min, max) em inteiros (reais)
    if not raw:
        return None
    nums = re.findall(r"\d+(?:[.,]\d+)?\s*[kK]?", str(raw))

    def parse_number_token(tok: str):
        t = tok.strip().lower().replace('.', '').replace(',', '.')
        scale = 1
        if t.endswith('k'):
            scale = 1000
            t = t[:-1]
        try:
            v = float(t)
        except Exception:
            return None
        return int(v * scale)

    parsed = [parse_number_token(t) for t in nums if parse_number_token(t) is not None]
    text_lower = str(raw).lower()
    thousand_hint = ('mil' in text_lower) or ('k' in text_lower)

    if len(parsed) >= 2:
        vmin, vmax = parsed[0], parsed[1]
        if vmin is None or vmax is None:
            return None
        if vmin < 1000 and vmax < 1000 and thousand_hint:
            vmin *= 1000
            vmax *= 1000
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return int(vmin), int(vmax)
    elif len(parsed) == 1:
        vmax = parsed[0]
        if vmax is None:
            return None
        if vmax < 1000 and thousand_hint:
            vmax *= 1000
        return 0, int(vmax)
    else:
        return None


def main():
    local, valor = get_user_input()

    interval = parse_valor_interval(valor)
    if interval is None:
        print("Intervalo de valor inválido. Informe algo como '30-50' ou '30000-50000' ou apenas '50000'.")
        return
    valor_min, valor_max = interval

    # Pegar todos os carros da localidade (passar um máximo alto) e filtrar localmente por intervalo
    cars_all = get_cars_from_sources(local, '99999999')
    if isinstance(cars_all, list) and cars_all and "erro" in cars_all[0]:
        print(cars_all[0]["erro"])
        return

    cars = []
    for c in cars_all:
        try:
            preco = int(c.get('preco', 0) or 0)
        except Exception:
            preco = 0
        if valor_min <= preco <= valor_max:
            cars.append(c)
    if not cars:
        print("Nenhum carro encontrado dentro do intervalo informado.")
        return

    recomendacao = analyze_cars(cars)
    format_output(cars, recomendacao)

if __name__ == "__main__":
    main()