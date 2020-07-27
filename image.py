import cv2
import os
import numpy as np
filename = '5205_boss.pgm'

def run(filename):
    path = r'D:\steganalysis_various_qtables\data\spatial\val'

    cover = cv2.imread(os.path.join(path, 'cover', filename))[...,0]
    mask = np.zeros([256, 256], dtype='float32')

    mask[15:15 + 235, 13:13 + 235] = 1
    cv2.imwrite('./im/mask.png', (mask*255).astype('uint8'))
    cv2.imwrite('./im/cover.png', cover)

    algo = ['s-uniward']

    for al in algo:
        for bp in [0.4]:
            stego_im =  cv2.imread(os.path.join(path, f'stego/{bp}/{al}', filename))[...,0]
            cv2.imwrite(f'im/{al}_{bp}.png', stego_im)
            diff = cover/1-stego_im/1
            diff[diff==0]=0
            cv2.imwrite(f'im/{al}_{bp}_diff.png', ((diff + 1)*127.5).astype('uint8'))

            masked = diff * (1-mask)
            lamb = masked.sum()/diff.sum()
            lam2 = (masked!=0).sum() / (diff!=0).sum()
            print(f'{al}_{bp}_{masked.sum()}_{diff.sum()}_{lamb}_{lam2}')
            cv2.imwrite(f'im/{al}_{bp}_masked1.png', ((diff * (1-mask) + 1)*127.5).astype('uint8'))
            cv2.imwrite(f'im/{al}_{bp}_masked2.png',  ((diff * mask + 1)*127.5).astype('uint8'))

if __name__ == '__main__':
    run(filename)
