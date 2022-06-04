import cv2 #import library pada camera virtual
import mediapipe as mp #import library untuk membuat line virtual
import time # import library fungsi waktu


class handDetector(): #pembuatan class dari fungsi deteksi tangan
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5): #pendefinisian dari program untuk jumlahtanagn, sensitivitas dari deteksi tangan
        self.mode = mode #inisialisasi variabel mode
        self.maxHands = maxHands #inisialisasi variabel maxHands
        self.detectionCon = detectionCon #inisialisasi variabel detectionCon
        self.trackCon = trackCon #inisialisasi variabel trackCon

        self.mpHands = mp.solutions.hands #untuk membuat garis tangan 
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon) #pada saat create maka akan menginisialisasi dari variabel Hands
        self.mpDraw = mp.solutions.drawing_utils #untuk menampilkan garis tangannya ke layar kamera
        
    def findHands(self, img, draw=True): #pendefinisian fungsi findHands untuk listid dan koordinat
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #inisialisasi dari color image hands pada Open-CV
        self.results = self.hands.process(imgRGB) #inisialisasi dari hasil pembacaan garis tangan
           # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks: #pengkondisian
            for handLms in self.results.multi_hand_landmarks: #perulangan program dari kondisi hasil pada garis tangan
                if draw:  #pengkondisian
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS) #inisialisasi nilai dari tangkapan garis tangan kamera, maka
        return img #pengembalian nilai ke variabel img
        
    def findPosition(self, img, handNo=0, draw=True):

        lmList = [] #variabel lmList yang berisi kosong
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] #myhand berisi koordinat
            for id, lm in enumerate(myHand.landmark): #enumerate untuk mengambil id
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) #menentukan posisi cx dan cy
                # print(id, cx, cy)
                lmList.append([id, cx, cy]) #mengisi matriks imList dengan id, cx, cy
            if draw:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList #mengembalikan fungsi lmList
        
def main(): #Blok kode utama
    pTime = 0 #inisialisasi variabel pTime
    cTime = 0 #inisialisasi variabel cTime
    cap = cv2.VideoCapture(1) #inisialisasi variabel cap dari nilai Open-CV
    detector = handDetector() #inisialisasi variabel detector
    while True: #pengkondisian program jika benar, maka
        success, img = cap.read() #read akan diinisalisasikan ke variabel succes, img
        img = detector.findHands(img) #inisisalisasi dari varaibel img
        lmList = detector.findPosition(img) #inisialisasi dari variabel lmlis
        if len(lmList) != 0: #pengkondisian jika len dengan parameter lmlist tidak sama dengan 0, maka
            print(lmList[4]) #menampilkan lmlist dengan nilai indeks 4

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img) #untuk menampilkan gambar di layar
        cv2.waitKey(1) #fungsi untuk mematikan open-cv secara otomatis


if __name__ == "__main__": #pemanggilan dari fungsi blok kode utama
    main()