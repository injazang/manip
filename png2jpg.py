import os
import glob
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing

def saveanddelete(imdir):
    if not imdir.endswith('.png'):
        return
    filename = os.path.basename(imdir).split('.')[0]
    if imdir.contains('train'):
        mode='train'
    elif imdir.contains('test'):
        mode = 'test'
    else:
        mode = 'val'
    newfilename = os.path.join('../dfdc_jpg', mode, filename+'.jpg')
    with Image.open(imdir) as im:
        im.save(newfilename, 'jpeg', quality=95)
    os.remove(imdir)
    return

def main(dir):
    num_cores = multiprocessing.cpu_count()
    images = glob.glob('../dfdc/*/*.png')
    results = Parallel(n_jobs=num_cores)(delayed(saveanddelete)(image) for image in images)

if __name__ == '__main__':
    main('../dfdc', '../dfdc_jpg')