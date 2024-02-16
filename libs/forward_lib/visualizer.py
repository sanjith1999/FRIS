import matplotlib.pyplot as plt
import torch


# Showing planes of a 3D Tensor
def show_planes(outs, title, N_z=16):  # outs.shape: [Nz, Nx, Ny]
    z_planes = range(0, N_z, max(N_z//8, 1))
    plt.figure(figsize=(20, 2))
    for i, z_idx in enumerate(z_planes):
        plt.subplot(1, len(z_planes), i+1)
        plt.imshow(outs[z_idx], vmin=outs.min(), vmax=outs.max())
        plt.axis('off')
        plt.title(f'z: {z_idx}')

    plt.subplots_adjust(top=0.73)
    plt.suptitle(f'{title} ')
    plt.show()



# Showing specified planes of a 3D Tensor
def show_planes_z(outs, title, z_planes):  # outs.shape: [Nz, Nx, Ny]
    plt.figure(figsize=(20, 1.8))
    for i, z_idx in enumerate(z_planes):
        plt.subplot(1, len(z_planes), i+1)
        plt.imshow(outs[z_idx], vmin=outs.min(), vmax=outs.max())
        plt.axis('off')
        plt.title(f'z: {z_idx}')

    plt.subplots_adjust(top=0.73)
    plt.suptitle(f'{title}')
    plt.show()


# Showing 2D Images
def show_image(image, title='', fig_size=(5, 5)):
    plt.figure(figsize=fig_size)
    plt.imshow(image, vmax=image.max(), vmin=image.min())
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


# Function to display images in a grid
def show_images(images, titles=None, cols=4, figsize=(12, 6)):
    rows = len(images) // cols + (len(images) % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if titles is None:
        titles = [f"Pattern: {i+1}" for i in range(len(images))]

    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axes.flatten()[i]
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Function to visualize 3D object
def vis_3d(image, title = "3D Object", elev_ang = 10, azim_ang = 40, fig_size = (4,4)):
    image = (image - image.min())/(image.max()-image.min())
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = image.shape
    colors = plt.cm.jet(image)

    ax.voxels(image, facecolors=colors, alpha= .5)

    ax.set_xlabel('Z'), ax.set_ylabel('X'), ax.set_zlabel('Y')
    ax.set_xlim(0, z), ax.set_ylim(0, x), ax.set_zlim(0, y)
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.view_init(elev=elev_ang, azim=azim_ang)
    ax.set_box_aspect([z, y, x])
    plt.show()


# Checking all the values of two vectors are equal with a specified tolerance
def compare_two_vectors(vec1, vec2, tolerance = 1e-6):
    are_approx_equal = torch.allclose(vec1, vec2, atol=tolerance)

    if are_approx_equal:
        print("The vectors are approximately the same.")
    else:
        print("The vectors are different.")

# Visual comparision of two one dimensional vectors
def visualize_vectors(V,cols = 2,  titles=None, fig_size = (12,6), same_fig = True, top_adjust = 2.5):
    n_vectors = len(V)
    if titles is None:
        titles = [f"vector: {i+1}" for i in range(n_vectors)]
    
    if same_fig:
        for i, (vector, title) in enumerate(zip(V, titles)):
            plt.plot(vector, label = title, alpha=.8)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Comparison of Vectors')

    else: 
        rows = n_vectors // cols + (n_vectors % cols > 0)
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)


        for i, (vector, title) in enumerate(zip(V, titles)):
            ax = axes.flatten()[i]
            ax.plot(vector, alpha=.8)
            ax.set_title(title)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(alpha=.8)
        plt.subplots_adjust(top=top_adjust)
    plt.show()


# Visual comparision of two or more planes
def visual_3Dcomparision(X, n_comparision = 2, Nz = 3, n_rows=1,fig_size = (18,4),titles=None):
    n_cols = max(Nz//n_rows,1)

    fig, big_axes = plt.subplots( figsize=fig_size , nrows=n_rows, ncols=n_cols, sharey=True) 
    for row, big_ax in enumerate(big_axes.flatten(), start=1):
        big_ax.set_title("Plane %s \n" % row, fontsize=16)
        big_ax.axis('off')
        big_ax._frameon = False
    for i in range(1, n_rows*n_cols+1):
        for j in range(n_comparision):
            ax = fig.add_subplot(n_rows,n_cols*n_comparision,i*n_comparision+j-(n_comparision-1))
            ax.imshow(X[j][i-1].abs())
            if titles and len(titles)>j:
                ax.set_title(titles[j])
            ax.set(xticks=[], yticks=[])
    fig.set_facecolor('w')
    plt.tight_layout()


def visualize_SSIM(measures, x_values=None, title = "SSIM Measure", y_label = 'SSIM Score', x_label = "Planes", labels=None,set_legend = False):
    plt.figure(figsize=(6, 6))  # Adjust the figure size if needed
    if not labels:
        labels = [f"Measure {i}" for i in range(len(measures))]
    for i, measure in enumerate(measures):
        if x_values and len(x_values)>i:
            plt.plot(x_values, measure, marker='o', linestyle='-', label = labels[i],alpha = .8)
        else:
            plt.plot(measure, marker='o', linestyle='-', label =labels[i], alpha=.8)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha = .6)
    if set_legend:
        plt.legend()
    plt.show()