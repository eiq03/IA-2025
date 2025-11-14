def get_user_input():
    local = input("Digite o local para busca: ")
    valor = input("Digite o intervalo de valor (ex: 30-50 mil ou 30000-50000). Para apenas máximo, digite 50000: ")
    return local, valor