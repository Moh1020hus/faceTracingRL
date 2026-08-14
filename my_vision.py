import os
import sys

import cv2
import face_recognition

REFERENCE_IMAGE = os.environ.get("REFERENCE_IMAGE", "me.jpg")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
SCALE = 0.25  # Detect on a quarter-size frame, then map boxes back to full res.
TOLERANCE = 0.6


def load_reference_encoding(path):
    if not os.path.exists(path):
        sys.exit(f"Reference image not found: {path}")

    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        # Indexing [0] here used to raise a bare IndexError.
        sys.exit(f"No face found in {path}. Use a clear, front-facing photo.")
    if len(encodings) > 1:
        print(f"Warning: {len(encodings)} faces in {path}, using the first one.")
    return encodings[0]


def main():
    print("Loading reference face...")
    my_encoding = load_reference_encoding(REFERENCE_IMAGE)

    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}.")

    print("Vision System Active. Looking for YOU...")
    inv_scale = int(1 / SCALE)

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Camera returned no frame - stopping.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            # face_recognition needs a contiguous RGB array; cvtColor gives one.
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            target_found = False
            height, width = frame.shape[:2]

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                distance = face_recognition.face_distance([my_encoding], face_encoding)[0]
                is_target = distance <= TOLERANCE

                # Scale the box back up to the full-resolution frame.
                top *= inv_scale
                right *= inv_scale
                bottom *= inv_scale
                left *= inv_scale

                if is_target:
                    target_found = True
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"TARGET (Dist: {distance:.2f})", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2

                    # Error is 0.0 at frame center, -1.0 at the left/top edge.
                    error_x = (center_x - (width / 2)) / (width / 2)
                    error_y = (center_y - (height / 2)) / (height / 2)

                    face_area = (bottom - top) * (right - left)
                    area_norm = face_area / (width * height)

                    print(
                        f"SENDING TO DRONE -> ErrorX: {error_x:.2f}, "
                        f"ErrorY: {error_y:.2f}, Area: {area_norm:.2f}"
                    )
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(
                        frame, "Unknown", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )

            if not target_found:
                print("Target Lost - Hovering...")

            cv2.imshow("Drone Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Ran only on the clean exit path before, so a crash held the camera open.
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
