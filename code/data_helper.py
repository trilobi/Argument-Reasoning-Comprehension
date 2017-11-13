import sys
import numpy as np

def iter_batch(batch_size, num_epochs, *args):
	args = list(args)
	assert len(args) > 0
	lens = [len(x) for x in args]
	current_idx = 0
	current_epoch = 1
	data_size = lens[0]
	while current_epoch <= num_epochs:
		if current_idx + batch_size <= data_size:
			end = current_idx + batch_size
			yield current_epoch, [x[current_idx: end] for x in args]
		else:
			end = (current_idx + batch_size) % data_size
			part1 = [x[current_idx: ] for x in args]
			part2 = [x[: end] for x in args]
			data_batch = []
			for i in range(len(args)):
				try:
					x = np.vstack((part1[i], part2[i])) 
				except:
					x = np.concatenate((part1[i], part2[i]))
				data_batch.append(x)
			yield current_epoch, data_batch
		if current_idx + batch_size < data_size:
			current_idx += batch_size
		else:
			current_idx = (current_idx + batch_size) % data_size
			current_epoch += 1
if __name__ == "__main__":
	pass
