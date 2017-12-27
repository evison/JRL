import os,sys
import gzip

review_file = sys.argv[1]
output_path = sys.argv[2]
min_count = int(sys.argv[3])
output_path += 'min_count' + str(min_count) + '/'
if not os.path.exists(output_path):
	os.makedirs(output_path)

#read all words, users, products
word_count_map = {}
user_set = set()
product_set = set()
with gzip.open(review_file, 'r') as g:
	for l in g:
		l = eval(l)
		user = l['reviewerID']
		product = l['asin']
		review_text = l['reviewText']
		summary = l['summary']
		user_set.add(user)
		product_set.add(product)
		for term in review_text.strip().split(' '):
			if term not in word_count_map:
				word_count_map[term] = 0
			word_count_map[term] += 1
		for term in summary.strip().split(' '):
			if term not in word_count_map:
				word_count_map[term] = 0
			word_count_map[term] += 1

#filter vocabulary by min_count
delete_key = set()
for key in word_count_map:
	if word_count_map[key] < min_count:
		delete_key.add(key)
#output word, user, product indexes
word_list = list(set(word_count_map.keys()) - delete_key)
with gzip.open(output_path + 'vocab.txt.gz','w') as fout:
	for word in word_list:
		fout.write(word + '\n')
user_list = list(user_set)
with gzip.open(output_path + 'users.txt.gz','w') as fout:
	for user in user_list:
		fout.write(user + '\n')
product_list = list(product_set)
with gzip.open(output_path + 'product.txt.gz','w') as fout:
	for product in product_list:
		fout.write(product + '\n')

#read and output indexed reviews
def index_set(s):
	i = 0
	s_map = {}
	for key in s:
		s_map[key] = str(i)
		i += 1
	return s_map
word_map = index_set(word_list)
user_map = index_set(user_list)
product_map = index_set(product_list)
with gzip.open(output_path + 'review_text.txt.gz', 'w') as fout_text, gzip.open(output_path + 'review_u_p.txt.gz', 'w') as fout_u_p:
	with gzip.open(output_path + 'review_id.txt.gz', 'w') as fout_id:
		with gzip.open(review_file, 'r') as g:
			index = 0
			for l in g:
				l = eval(l)
				user = l['reviewerID']
				product = l['asin']
				review_text = l['reviewText']
				summary = l['summary']
				count_words = 0
				for term in summary.strip().split(' '):
					if term in word_map:
						fout_text.write(word_map[term] + ' ')
						count_words += 1
				for term in review_text.strip().split(' '):
					if term in word_map:
						fout_text.write(word_map[term] + ' ')
						count_words += 1

				if count_words > 0:
					fout_text.write('\n')
					fout_u_p.write(user_map[user] + ' ' + product_map[product] + '\n')
					fout_id.write('line_' + str(index) + '\n')
				index += 1
				
				
		
