def format_output(cars, recomendacao):
    print("\n=== RESULTADOS ENCONTRADOS ===")
    for c in cars:
        print(f"{c['modelo']} | R${c['preco']} | {c['local']}")
    
    print("\n=== RECOMENDAÇÃO DA IA ===")
    print(recomendacao)