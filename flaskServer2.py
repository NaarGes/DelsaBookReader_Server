from flask import Flask, request, send_file, send_from_directory
import cv2
import KDStorage

# Server configurations.
ip, port = '192.168.43.133', 5000

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

        book = []
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

            book.append(len(good))

        max_book = max(book)
        index_max_book = book.index(max_book)
        #
        # if max_book < 100:  # not a book picture
        #     pagenumber = 0
        # else:
        pagenumber = index_max_book + 1
        print('book good matches: {}'.format(book))
        print('match page: {}'.format(index_max_book+1))

    print('returning response: http://{ip}:{port}/statics/miss-brush/woman/{page_number}.ogg'.format(ip=ip, port=port, page_number=pagenumber))
    return "http://{ip}:{port}/statics/miss-brush/woman/{page_number}.ogg".format(ip=ip, port=port, page_number=pagenumber)

# Run the app :)
if __name__ == '__main__':
    app.run(
        host=ip,
        port=port
    )
