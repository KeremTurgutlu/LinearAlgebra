from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import glob

# RETURNS IMAGE WITH SPECIFIED SINGULAR VALUES
def SVD_Image(image_path, k = None, full_rank = None, full_column = None,  plot = "bar"):

    if type(image_path) == str:
        # READ IMAGE AND COMPUTE SYMMETRICAL MATRICES
        img = Image.open(image_path)
        img = img.convert("L")
        ncols = img.size[0]
        nrows = img.size[1]
        A = np.asarray(img.getdata()).reshape(nrows, ncols)
    else:
        A = image_path
        ncols = image_path.shape[1]
        nrows = image_path.shape[0]


    Q1 = A.dot(A.T)
    Q2 = A.T.dot(A)


    # FIND RANK OF A
    r = np.linalg.matrix_rank(A)
    if full_rank: k = r
    elif full_column: k = ncols

    # FIND V AND SINGULAR VALUES
    sigma_2, v = np.linalg.eig(Q2)
    singular_vals = np.sqrt(sigma_2)
    diagonal_singvals = np.diag(singular_vals)


    # COMPUTE U
    u = np.dot(A, v).dot(np.linalg.inv(diagonal_singvals))

    A2 = np.zeros(ncols*nrows).reshape(nrows, ncols)

    # RECONSTRUCT IMAGE

    for i in range(k):
      A2 +=  singular_vals[i].real*(np.outer(u[:, i].real, v[:, i].real))

    img = Image.fromarray(np.uint8(A2))
    img.show()

    # PLOT SINGULAR VALUES
    if plot == "bar":
        plt.bar(np.arange(k)+1, singular_vals[:k])
        plt.xlabel("Singular Value Rank")
        plt.ylabel("Singular Value")
        plt.title('Top %s Singular Values' %k)
        plt.show()
    elif plot == "line":
        plt.plot(np.arange(k) + 1, singular_vals[:k])
        plt.xlabel("Singular Value Rank")
        plt.ylabel("Singular Value")
        plt.title('Top %s Singular Values' % k)
        plt.show()

# PLOTS SINGULAR VALUE DISTRIBUTIONS FOR GIVEN LIST OF IMAGES
def SVD_Plot(imagepath_list):
    names = []
    sing_vals = []
    i = 0
    for image_path in imagepath_list:
        # READ IMAGE AND COMPUTE SYMMETRICAL MATRICES
        if type(image_path) == str:
            img = Image.open(image_path)
            img = img.convert("L")
            ncols = img.size[0]
            nrows = img.size[1]
            A = np.asarray(img.getdata()).reshape(nrows, ncols)
            names.append(image_path.split("/")[1].split(".")[0])
        else:
            i += 1
            A = image_path
            ncols = image_path.shape[1]
            nrows = image_path.shape[0]
            names.append("random %s" %i)
        Q1 = A.dot(A.T)
        Q2 = A.T.dot(A)


        # FIND V AND SINGULAR VALUES
        sigma_2, v = np.linalg.eig(Q2)
        singular_vals = np.sqrt(sigma_2)
        sing_vals.append(singular_vals)



    for i in range(len(imagepath_list)):
        val = sing_vals[i]
        maxi = max(val)
        mini = min(val)
        normalized_val = (val - mini) / (maxi - mini)
        plab.plot(normalized_val,label = names[i])

    plab.title("Singular Value Distributions")
    plab.xlabel("Singular Value Rank")
    plab.ylabel("Singular Value")
    plab.xlim(0, 30)
    plab.ylim(0, 1)
    plab.legend()
    plab.show()

def SVD_Image2(image_path, k, plot = "bar"):

    if type(image_path) == str:
        # READ IMAGE AND COMPUTE SYMMETRICAL MATRICES
        img = Image.open(image_path)
        img = img.convert("L")
        ncols = img.size[0]
        nrows = img.size[1]
        A = np.asarray(img.getdata()).reshape(nrows, ncols)
    else:
        A = image_path
        ncols = image_path.shape[1]
        nrows = image_path.shape[0]

    Q1 = A.dot(A.T)
    Q2 = A.T.dot(A)

    # FIND RANK OF A
    r = np.linalg.matrix_rank(A)

    # FIND V AND SINGULAR VALUES
    sigma_2, v = np.linalg.eigh(Q2)
    singular_vals = np.sqrt(list(reversed(sigma_2))[:r])
    v = np.flip(v, 1)
    diagonal_singvals = np.diag(singular_vals)


    # COMPUTE U
    u = np.dot(A, v[:, :r]).dot(np.linalg.inv(diagonal_singvals))
    print(u.shape)
    print(v.shape)
    A2 = np.zeros(ncols*nrows).reshape(nrows, ncols)

    # RECONSTRUCT IMAGE
    for i in range(k):
      A2 +=  singular_vals[i].real*(np.outer(u[:, i].real, v[:, i].real))

    A2 =  np.dot(u[:, :k], diagonal_singvals[:k, :k]).dot(v.T[:k, :])

    img = Image.fromarray(np.uint8(A2))
    img.show()

    # PLOT SINGULAR VALUES
    if plot == "bar":
        plt.bar(np.arange(k)+1, singular_vals[:k])
        plt.xlabel("Singular Value Rank")
        plt.ylabel("Singular Value")
        plt.title('Top %s Singular Values' %k)
        plt.show()
    elif plot == "line":
        plt.plot(np.arange(k) + 1, singular_vals[:k])
        plt.xlabel("Singular Value Rank")
        plt.ylabel("Singular Value")
        plt.title('Top %s Singular Values' % k)
        plt.show()

SVD_Image2("me.jpg", 20)