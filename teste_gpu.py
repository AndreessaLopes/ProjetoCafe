import torch
print(f"Tem GPU? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nome: {torch.cuda.get_device_name(0)}")
else:
    print("ERRO: O Python ainda não está vendo sua placa!")