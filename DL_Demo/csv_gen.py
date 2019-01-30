import csv
import random

timespan = 1000
with open('data.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	
	total_ads = 0
	decay_coefficient = 0.8
	for i in range(timespan):
		ad_num = random.randint(0, 10)
		donation_amout = random.randint(0, int(total_ads))

		# Foramt: day_index, posted advertisment number, amout of donation
		writer.writerow([i, ad_num, donation_amout])

		total_ads = ad_num + decay_coefficient * decay_coefficient

