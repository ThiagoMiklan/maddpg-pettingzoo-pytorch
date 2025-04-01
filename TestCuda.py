import torch
print(torch.cuda.is_available())  # Deve retornar True
print(torch.cuda.device_count())  # Deve retornar pelo menos 1
print(torch.cuda.current_device())  # Deve retornar 0
print(torch.cuda.get_device_name(0))  # Deve exibir "NVIDIA GeForce GTX 1650"