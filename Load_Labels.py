import xml.etree.ElementTree as ET
import numpy as np

num_image = "004"

tree = ET.parse("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + num_image + ".xml")

def give_label(tree):
	labels = []

	for elem in tree.iter():

		if 'object' in elem.tag or 'part' in elem.tag:
			for attr in list(elem):

				aux = []

				if 'name' in attr.tag:
					name = attr.text
					aux.append([0, 1])

				if 'bndbox' in attr.tag:
					for dim in list(attr):

						if 'xmin' in dim.tag:
							xmin = int(round(float(dim.text)))
						if 'ymin' in dim.tag:
							ymin = int(round(float(dim.text)))
						if 'xmax' in dim.tag:
							xmax = int(round(float(dim.text)))
						if 'ymax' in dim.tag:
							ymax = int(round(float(dim.text)))

					if name[0] == "R":
						# presencia de objeto
						aux.append(1)

						# dimenisons de la bounding box
						x = xmin + (xmax - xmin) / 2
						y = ymin + (ymax - ymin) / 2

						w = xmax - xmin
						h = ymax - ymin

						aux.append(x)
						aux.append(y)
						aux.append(w)
						aux.append(h)

						# clase a la que pertenece
						aux.append(0)
						aux.append(0)
						aux.append(1)

					if name[0] == "W":
						# presencia de objeto
						aux.append(1)

						# dimenisons de la bounding box
						x = xmin + (xmax - xmin) / 2
						y = ymin + (ymax - ymin) / 2

						w = xmax - xmin
						h = ymax - ymin

						aux.append(x)
						aux.append(y)
						aux.append(w)
						aux.append(h)

						# clase a la que pertenece
						aux.append(0)
						aux.append(1)
						aux.append(0)

					if name[0] == "P":
						# presencia de objeto
						aux.append(1)

						# dimenisons de la bounding box
						x = xmin + (xmax - xmin) / 2
						y = ymin + (ymax - ymin) / 2

						w = xmax - xmin
						h = ymax - ymin

						aux.append(x)
						aux.append(y)
						aux.append(w)
						aux.append(h)

						# clase a la que pertenece
						aux.append(1)
						aux.append(0)
						aux.append(0)

			labels.append(aux)
	return np.asarray(labels)


def labels_total(label):
	n_grid_cells = 10

	l_total = []
	x = 640 / n_grid_cells
	y = 480 / n_grid_cells


	for j in range(n_grid_cells):
		for i in range(n_grid_cells):

			alpha = 0
			b = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])

			dx1 = (j) * x
			dy1 = (i) * y
			dx2 = (j+1) * x
			dy2 = (i+1) * y


			for k in range(len(label)):


				a = label[k]

				if a[1] > dx1 and a[1] < dx2 and a[2] > dy1 and a[2] < dy2:

					alpha = 1

					## l_total.append(a)

					break

			if alpha==1:
				l_total.append(a)
			else:
				l_total.append(b)

	return l_total


def abrir_labels(n_imag):
	path = "000"

	total = []

	for i in range(n_imag):
		path1 = int(path)
		path1 += 1
		path2 = str(path1)

		tree = ET.parse("C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + num_image + ".xml")

		aa = np.asarray(give_label(tree))
		bb = labels_total(aa)

		total.append(bb)


		total.append(bb)
		total.append(bb)
		total.append(bb)
		total.append(bb)
		total.append(bb)
		total.append(bb)
		total.append(bb)
		total.append(bb)
		total.append(bb)




		path2 = path

	return np.asarray(total)
