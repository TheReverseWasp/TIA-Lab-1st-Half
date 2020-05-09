import json

def main():
	json_data ={}
	learning_rates = [0.01, 0.05, 0.1, 0.2, 0.4]
	J_step = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
	fixed_dataset_names = ["home", "ocean", "petrol"]
	with open('../Resultados/Exp2/Resultado2.json') as json_file:
		json_data= json.load(json_file)
	separated_data = []
	try:
		for i in range(len(fixed_dataset_names)):
			separated_data.append(fixed_dataset_names[i])
			separated_data.append("train")
			for j in learning_rates:
				separated_data.append([[j], json_data[str(i)+"-"+str(j)+":train"]])
			separated_data.append("test")
			for j in learning_rates:
				separated_data.append([[j], json_data[str(i)+"-"+str(j)+":test"]])
	except:
		print("faltan datasets originales ")
	try:
		separated_data.append("other")
		separated_data.append("train")
		for j in learning_rates:
			separated_data.append([[j], json_data[str(i)+"-"+str(j)+":train"]])
		separated_data.append("test")
		for j in learning_rates:
			separated_data.append([[j], json_data[str(i)+"-"+str(j)+":test"]])
	except:
		print("faltan datasets provicionales ")
	with open('../Resultados/Exp2/transcripted.txt', 'w') as transcripted:
		for i in separated_data:
			transcripted.write(str(i) + "\n")

if __name__ == "__main__":
	main()