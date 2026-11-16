import cv2

# Use DirectShow instead of Windows MSMF
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Camera opened successfully.")

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
