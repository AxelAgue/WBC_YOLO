import xml.etree.ElementTree as ET
import numpy as np

"""
PATHS:

OMEN: "C:/Users/axel_/Desktop/Varios/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + path + ".xml"
SAMIN: "/home/oem/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + path + ".xml"
GUAMAN:

"""

# num_image = 1


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





	return np.asarray(l_total)

def correct_label(label):

	lenght = label.shape[0]
	total = []
	for i in range(lenght):
		alpha = label[i]

		aa = label[i]

		if aa[1] < 80:
			alpha[1] = 0
		else:
			alpha[1] = aa[1] - 80

		total.append(alpha)

	bb = np.asarray(total)
	aa = bb.reshape(lenght, 8)
	return aa



def augmented_labels_90(label, n_grid_cells):
	total_label = []

	for i in range(n_grid_cells ** 2):

		alpha = label[i]
		aux = label[i]
		if aux[0] == 1:
			alpha[1] = 480 - aux[1]
			alpha[2] = aux[2]
			alpha[3] = aux[4]
			alpha[4] = aux[3]
		total_label.append(alpha)

	d = np.asarray(total_label)
	return d.reshape(100, 8)

def augmented_labels_180(label, n_grid_cells):
	total_label = []

	for i in range(n_grid_cells ** 2):

		alpha = label[i]
		aux = label[i]
		if aux[0] == 1:
			alpha[1] = aux[1]
			alpha[2] = 480 - aux[2]
			alpha[3] = aux[4]
			alpha[4] = aux[3]
		total_label.append(alpha)
	d = np.asarray(total_label)
	return d.reshape(100, 8)

def augmented_labels_270(label, n_grid_cells):
	total_label = []

	for i in range(n_grid_cells ** 2):

		alpha = label[i]
		aux = label[i]
		if aux[0] == 1:
			alpha[1] = 480 - aux[1]
			alpha[2] = 480 - aux[2]
			alpha[3] = aux[4]
			alpha[4] = aux[3]
		total_label.append(alpha)
	d = np.asarray(total_label)
	return d.reshape(100, 8)

def invertir_labels_x(label, n_grid_cells):

	total_label = []

	for i in range(n_grid_cells ** 2):

		alpha = label[i]
		aux = label[i]
		if aux[0] == 1:
			alpha[1] = 480 - aux[1]
		total_label.append(alpha)
	d = np.asarray(total_label)
	return d.reshape(100, 8)

def invertir_labels_y(label, n_grid_cells):

	total_label = []

	for i in range(n_grid_cells ** 2):

		alpha = label[i]
		aux = label[i]
		if aux[0] == 1:
			alpha[2] = 480 - aux[2]
		total_label.append(alpha)
	d = np.asarray(total_label)
	return d.reshape(100, 8)

"""
yy = give_label(tree)
aa = labels_total(10, yy)
bb = correct_label(aa)
prueba = invertir_labels_x(bb, 10)
"""

def abrir_labels(n_imag):
	path = "000"

	labels = []

	for i in range(n_imag):
		path1 = int(path)
		path1 += 1
		path2 = str(path1)

		tree = ET.parse("/home/oem/BCCD_Dataset/BCCD/Annotations/BloodImage_00" + path + ".xml")


		yy = give_label(tree)
		aa = labels_total(yy)
		bb = correct_label(aa)

		label_rotada1 = augmented_labels_90(bb, n_grid_cells=10)
		label_rotada2 = augmented_labels_180(bb, n_grid_cells=10)
		label_rotada3 = augmented_labels_270(bb, n_grid_cells=10)

		h_l1 = invertir_labels_x(label_rotada1, n_grid_cells=10)
		v_l1 = invertir_labels_y(label_rotada1, n_grid_cells=10)

		h_l2 = invertir_labels_x(label_rotada2, n_grid_cells=10)
		v_l2 = invertir_labels_y(label_rotada3, n_grid_cells=10)

		h_l3 = invertir_labels_x(label_rotada2, n_grid_cells=10)
		v_l3 = invertir_labels_y(label_rotada3, n_grid_cells=10)

		labels.append(bb)

		labels.append(label_rotada1)
		labels.append(label_rotada2)
		labels.append(label_rotada3)

		labels.append(h_l1)
		labels.append(v_l1)
		labels.append(h_l2)
		labels.append(v_l2)
		labels.append(h_l3)
		labels.append(v_l3)

		path2 = path

	return np.asarray(labels)


alpha = abrir_labels(4)

print(alpha.shape)