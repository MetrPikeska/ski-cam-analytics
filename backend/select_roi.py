"""
Nástroj pro interaktivní výběr ROI a Line Crossing.
Spusť: python select_roi.py
"""
import cv2
import sys
sys.path.insert(0, '.')
from app.ffmpeg_source import FFmpegSource
from app.config import STREAM_URL, FFMPEG_FPS, FRAME_WIDTH, FRAME_HEIGHT

# Globální proměnné
roi_points = []
line_points = []
mode = "roi"  # "roi" nebo "line"
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    global roi_points, line_points, frame_copy, mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "roi":
            roi_points.append((x, y))
            print(f"ROI bod {len(roi_points)}: ({x}, {y})")
            
            if len(roi_points) == 2:
                # Nakresli obdélník
                cv2.rectangle(frame_copy, roi_points[0], roi_points[1], (0, 255, 0), 2)
                print(f"\nROI_RECT = {roi_points[0] + roi_points[1]}")
                print("Zkopíruj tento řádek do config.py!\n")
        
        elif mode == "line":
            line_points.append((x, y))
            print(f"Line bod {len(line_points)}: ({x}, {y})")
            
            if len(line_points) == 2:
                # Nakresli čáru
                cv2.line(frame_copy, line_points[0], line_points[1], (255, 0, 0), 2)
                print(f"\nLINE_CROSSING = {line_points}")
                print("Zkopíruj tento řádek do config.py!\n")

def main():
    global frame_copy, mode, roi_points, line_points
    
    print("=== ROI & Line Crossing Selector ===")
    print("1. Stáhnu frame ze streamu...")
    
    # Získat frame
    ffmpeg = FFmpegSource(STREAM_URL, FFMPEG_FPS, FRAME_WIDTH, FRAME_HEIGHT)
    ffmpeg.start()
    
    frame = ffmpeg.get_frame(timeout=10)
    if frame is None:
        print("Chyba: Nepodařilo se získat frame!")
        return
    
    ffmpeg.stop()
    frame_copy = frame.copy()
    
    print("2. Otevírám okno...")
    print("\nINSTRUKCE:")
    print("- Klikni na 2 rohy pro ROI (zelený obdélník)")
    print("- Stiskni 'L' pro přepnutí na Line Crossing")
    print("- Klikni na 2 body pro Line (modrá čára)")
    print("- Stiskni 'R' pro reset")
    print("- Stiskni 'Q' pro ukončení")
    
    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", mouse_callback)
    
    while True:
        display = frame_copy.copy()
        
        # Zobraz aktuální body
        if mode == "roi" and len(roi_points) == 1:
            cv2.circle(display, roi_points[0], 5, (0, 255, 0), -1)
        elif mode == "line" and len(line_points) == 1:
            cv2.circle(display, line_points[0], 5, (255, 0, 0), -1)
        
        # Zobraz finální výsledky
        if len(roi_points) == 2:
            cv2.rectangle(display, roi_points[0], roi_points[1], (0, 255, 0), 2)
        if len(line_points) == 2:
            cv2.line(display, line_points[0], line_points[1], (255, 0, 0), 2)
        
        # Zobraz režim
        text = f"Mode: {mode.upper()} | ROI: {len(roi_points)}/2 | Line: {len(line_points)}/2"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("ROI Selector", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi_points = []
            line_points = []
            frame_copy = frame.copy()
            print("Reset!")
        elif key == ord('l'):
            mode = "line"
            print("Režim: Line Crossing")
        elif key == ord('o'):
            mode = "roi"
            print("Režim: ROI")
    
    cv2.destroyAllWindows()
    
    print("\n=== VÝSLEDKY ===")
    if len(roi_points) == 2:
        print(f"ROI_RECT = {roi_points[0] + roi_points[1]}")
    if len(line_points) == 2:
        print(f"LINE_CROSSING = {line_points}")

if __name__ == "__main__":
    main()
