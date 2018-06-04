from flask import Flask, request, send_file, send_from_directory
import cv2
import KDStorage

# Server configurations.
ip, port = '192.168.4.94', 5000

app = Flask(__name__, static_url_path='')


@app.route('/statics/<path:path>')
def statics(path):
    return send_from_directory('statics', path)


@app.route('/files/', methods=['POST'])
def files():
    with open('file.png', 'wb') as f:
        f.write(request.files['myFile'].read())

        # compare image
        sameimage = False
        pagenumber = 0
        Qimg = cv2.imread('file.png', 0)  # query image

        sift = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector
        Qkp, Qdes = sift.detectAndCompute(Qimg, None)  # find the keypoints and descriptors with SIFT
        bf = cv2.BFMatcher()  # BFMatcher with default params

        for i in range(1, 17):
            print('loading keypoints and descriptors of image {}...'.format(i))
            tkd = KDStorage.load('MissBrush{}.kd'.format(i))
            print('start matching with image {}...'.format(i))
            matches = bf.knnMatch(Qdes, tkd[1], k=2)
            # Apply ratio test
            good = []
            print('starting ratio test...')
            for m, n in matches:
                if m.distance < 0.65 * n.distance:
                    good.append([m])
            if len(good) / len(tkd[0]) > 0.1:
                pagenumber = i
                sameimage = True

            if sameimage is True:
                print('Image is equal to {}.jpg'.format(i))
                break
            elif i == 17:
                print('no match found')
            else:
                print('not equal to image {}'.format(i))

    print('returning response: http://{ip}:{port}/statics/miss-brush/woman/{page_number}.ogg'.format(ip=ip, port=port, page_number=pagenumber))
    return "http://{ip}:{port}/statics/miss-brush/woman/{page_number}.ogg".format(ip=ip, port=port, page_number=pagenumber)
    # return "http://{ip}:{port}/statics/miss-brush/man/doll-box-conjuring.mp3".format(ip=ip, port=port)

# Run the app :)
if __name__ == '__main__':
    app.run(
        host=ip,
        port=port
    )
