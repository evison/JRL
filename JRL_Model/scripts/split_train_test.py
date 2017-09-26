import os,sys
import gzip
import numpy as np

data_path = sys.argv[1]
sample_rate = float(sys.argv[2])

#build user-review map
user_review_map = {}
with gzip.open(data_path + 'review_u_p.txt.gz', 'r') as fin:
	index = 0
	for line in fin:
		arr = line.strip().split(' ')
		user = arr[0]
		if user not in user_review_map:
			user_review_map[user] = []
		user_review_map[user].append(index)
		index += 1

#generate train/test sets

test_review_idx = set()
for user in user_review_map:
	sample_number = int(len(user_review_map[user]) * sample_rate)
	#sample_number = 1 if sample_number < 1 else sample_number
	test_review_idx |= set(np.random.choice(user_review_map[user], sample_number , replace=False))

#read review_text and construct train/test sets
train_user_product_map = {}
test_user_product_map = {}
with gzip.open(data_path + 'train.txt.gz', 'w') as train_fout, gzip.open(data_path + 'test.txt.gz', 'w') as test_fout:
	with gzip.open(data_path + 'review_u_p.txt.gz', 'r') as info_fin, gzip.open(data_path + 'review_text.txt.gz', 'r') as text_fin:
		info_line = info_fin.readline()
		text_line = text_fin.readline()
		index = 0
		while info_line:
			arr = info_line.strip().split(' ')
			if index not in test_review_idx:
				train_fout.write(arr[0] + '\t' + arr[1] + '\t' + text_line.strip() + '\n')
				if int(arr[0]) not in train_user_product_map:
					train_user_product_map[int(arr[0])] = set()
				train_user_product_map[int(arr[0])].add(int(arr[1]))
			else:
				test_fout.write(arr[0] + '\t' + arr[1] + '\t' + text_line.strip() + '\n')
				if int(arr[0]) not in test_user_product_map:
					test_user_product_map[int(arr[0])] = set()
				test_user_product_map[int(arr[0])].add(int(arr[1]))
			index += 1
			info_line = info_fin.readline()
			text_line = text_fin.readline()

#read review_u_p and construct train/test id sets
with gzip.open(data_path + 'train_id.txt.gz', 'w') as train_fout, gzip.open(data_path + 'test_id.txt.gz', 'w') as test_fout:
	with gzip.open(data_path + 'review_u_p.txt.gz', 'r') as info_fin, gzip.open(data_path + 'review_id.txt.gz', 'r') as id_fin:
		info_line = info_fin.readline()
		id_line = id_fin.readline()
		index = 0
		while info_line:
			arr = info_line.strip().split(' ')
			if index not in test_review_idx:
				train_fout.write(arr[0] + '\t' + arr[1] + '\t' + str(id_line.strip()) + '\n')
			else:
				test_fout.write(arr[0] + '\t' + arr[1] + '\t' + str(id_line.strip()) + '\n')
			index += 1
			info_line = info_fin.readline()
			id_line = id_fin.readline()

for u_idx in test_user_product_map:
	test_user_product_map[u_idx] -= train_user_product_map[u_idx]


#output qrels
product_ids = []
with gzip.open(data_path + 'product.txt.gz', 'r') as fin:
	for line in fin:
		product_ids.append(line.strip())

user_ids = []
with gzip.open(data_path + 'users.txt.gz', 'r') as fin:
	for line in fin:
		user_ids.append(line.strip())

with open(data_path + 'test.qrels', 'w') as fout:
	for u_idx in test_user_product_map:
		user_id = user_ids[u_idx]
		for product_idx in test_user_product_map[u_idx]:
			product_id = product_ids[product_idx]
			fout.write(user_id + ' 0 ' + product_id + ' 1 ' + '\n')



