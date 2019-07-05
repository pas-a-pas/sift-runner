from PIL import Image
from pylab import *
import sift
import os

class SiftRunner:
    def __init__(self, path):
        """ Construct gets a path to images """

        self.images = SiftRunner.__get_images__(path)

    def process(self):
        """ Run sift in a batch for each image in the given path """

        for image in self.images:
            sift.process_image(image, SiftRunner.__get_sift_id__(image))

    def full_match(self, visualize=True):
        """ Match features in 2 images of all combinations """

        n = len(self.images)
        matchcounts = zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                scores = SiftRunner.match(self.images[i], self.images[j])
                count = sum(scores > 0)
                matchcounts[i, j] = count
                matchcounts[j, i] = matchcounts[i, j]

        if visualize:
            self.visualize(matchcounts)

        return matchcounts

    def match(image1, image2, visualize=False):
        """ Match features in 2 images """

        print('comparing ', image1, image2)
        locs1, desc1 = sift.read_features_from_file(SiftRunner.__get_sift_id__(image1))
        locs2, desc2 = sift.read_features_from_file(SiftRunner.__get_sift_id__(image2))
        scores = sift.match_twosided(desc1, desc2)

        if visualize:
            im1 = array(Image.open(image1).convert('L'))
            im2 = array(Image.open(image2).convert('L'))

            fig = figure()
            grid = GridSpec(2, 2)
            fig.add_subplot(grid[0, 0])
            sift.plot_feature(im1, locs1)
            fig.add_subplot(grid[0, 1])
            sift.plot_feature(im2, locs2)
            fig.add_subplot(grid[1, :])
            sift.plot_matches(im1, im2, locs1, locs2, scores, show_below=False)
            show()

        return scores

    def visualize(self, matchcounts):
        """ Visualize matched group.
        For each row in a table, an image is shown and  matched images are followed.
        Can check the match side by side by clicking a matched image. """

        n = len(self.images)
        ncols = 0
        for i in range(n):
            count = sum(matchcounts[i] > 0)
            if ncols < count:
                ncols = count

        fig = figure()
        for i in range(n):
            colindex = 1
            axes = fig.add_subplot(n, ncols, i * ncols + colindex)
            axes.label = {'image-id': i}
            imshow(SiftRunner.__get_thumbnail__(self.images[i]))
            axis('off')

            for j in range(n):
                if i != j and matchcounts[i][j] > 0:
                    colindex += 1
                    axes = fig.add_subplot(n, ncols, i * ncols + colindex)
                    axes.label = {'image-id': i, 'matched-image-id': j}
                    imshow(SiftRunner.__get_thumbnail__(self.images[j]))
                    axis('off')

        def onclick(event):
            if event.inaxes:
                if 'image-id' in event.inaxes.label and 'matched-image-id' in event.inaxes.label:
                    left = event.inaxes.label['image-id']
                    right = event.inaxes.label['matched-image-id']
                    SiftRunner.match(self.images[left], self.images[right], visualize=True)

        fig.canvas.mpl_connect('button_press_event', onclick)
        show()

    def __get_images__(path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".JPG")]

    def __get_thumbnail__(image):
        im = Image.open(image)
        im.thumbnail((100, 100))
        return im

    def __get_sift_id__(seed):
        return str(seed + '.sift')

path = 'samples/'
runner = SiftRunner(path)
runner.process()
runner.full_match()
