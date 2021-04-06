# Histogram utils

def histogram_plot(img_data=None):
  (x_data, y_data) = img_data if img_data else (x_train, y_train)

  fig = plt.figure(figsize=(12,5))

  plt.subplot(1,2,1)
  plt.hist(y_data, bins = range(int(y_data.min()), int(y_data.max() + 2)))
  plt.xticks(range(int(y_data.min()), int(y_data.max() + 2)))
  plt.title("y histogram")
  plt.subplot(1,2,2)
  plt.hist(x_data.flat, bins = range(int(x_data.min()), int(x_data.max() + 2)))
  plt.title("x histogram")
  plt.tight_layout()
  plt.show()

  hist, bins = np.histogram(y_data, bins = range(int(y_data.min()), int(y_data.max() + 2)))
  print('y histogram counts:', hist)

def histogram_label_plot(train_img_data=None, test_img_data=None):
  (x_train_data, y_train_data) = train_img_data if train_img_data else (x_train, y_train)
  (x_test_data, y_test_data) = test_img_data if test_img_data else (x_test, y_test)

  x_data_min = int(min(x_train_data.min(), x_test_data.min()))
  x_data_max = int(min(x_train_data.max(), x_test_data.max()))
  y_data_min = int(min(y_train_data.min(), y_test_data.min()))
  y_data_max = int(min(y_train_data.max(), y_test_data.max()))
  num_rows = y_data_max - y_data_min + 1
  
  fig = plt.figure(figsize=(12,12))

  plot_num = 1
  for lbl in range(y_data_min, y_data_max):
    plt.subplot(num_rows, 2 , plot_num)
    plt.hist(x_train_data[y_train_data.squeeze() == lbl].flat, bins = range(x_data_min, x_data_max + 2))
    plt.title("x train histogram - label %s" % lbl)
    plt.subplot(num_rows, 2 , plot_num + 1)
    plt.hist(x_test_data[y_test_data.squeeze() == lbl].flat, bins = range(x_data_min, x_data_max + 2))
    plt.title("x test histogram - label %s" % lbl)
    plot_num += 2

  plt.tight_layout(pad=0)
  plt.show()
