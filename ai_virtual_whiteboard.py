import cv2
import mediapipe as mp
import numpy as np
import time
import math 

def is_finger_open(tip_id, pip_id, hand_landmarks, wrist_landmark):
    tip_distance = math.hypot(hand_landmarks[tip_id].x - wrist_landmark.x, hand_landmarks[tip_id].y - wrist_landmark.y)
    pip_distance = math.hypot(hand_landmarks[pip_id].x - wrist_landmark.x, hand_landmarks[pip_id].y - wrist_landmark.y)
    return tip_distance > pip_distance

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mediapipe_hands = mp.solutions.hands
hands_tracker = mediapipe_hands.Hands(
    max_num_hands=1, 
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mediapipe_drawing = mp.solutions.drawing_utils

drawing_canvas = np.zeros((720, 1280, 3), np.uint8)
previous_x = 0
previous_y = 0
smoothed_x = 0
smoothed_y = 0  
smoothed_thickness = 10 

canvas_history_stack = []
is_user_drawing = False
undo_hover_timer = 0
clear_board_timer = 0 
was_in_shape_mode = False
last_drawn_circle = None

available_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] 
color_rectangles = [(20, 20, 120, 100), (140, 20, 240, 100), (260, 20, 360, 100), (380, 20, 480, 100)]
current_draw_color = available_colors[1] 
eraser_brush_thickness = 50

is_camera_showing = True
is_grid_mode_active = False
is_measure_mode_active = False
is_privacy_mode_active = False 
is_laser_mode_active = False 

display_status_text = ""
display_status_timer = 0

while True:
    read_success, camera_frame = video_capture.read()
    if not read_success:
        break

    camera_frame = cv2.resize(camera_frame, (1280, 720))
    camera_frame = cv2.flip(camera_frame, 1)
    
    clean_camera_feed = camera_frame.copy() 
    
    frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
    hand_tracking_results = hands_tracker.process(frame_rgb)
    frame_height, frame_width, frame_channels = camera_frame.shape
    
    index_finger_x = 0
    index_finger_y = 0 

    if is_privacy_mode_active:
        privacy_mask = np.zeros((frame_height, frame_width), dtype=np.uint8) 
        if hand_tracking_results.multi_hand_landmarks:
            for hand_landmarks_data in hand_tracking_results.multi_hand_landmarks:
                hand_points = []
                for landmark_point in hand_landmarks_data.landmark:
                    hand_points.append([int(landmark_point.x * frame_width), int(landmark_point.y * frame_height)])
                
                hand_points_array = np.array(hand_points, dtype=np.int32)
                convex_hull_boundary = cv2.convexHull(hand_points_array)
                cv2.drawContours(privacy_mask, [convex_hull_boundary], -1, 255, cv2.FILLED)
            
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            privacy_mask = cv2.dilate(privacy_mask, dilation_kernel, iterations=1)
            blurred_privacy_mask = cv2.GaussianBlur(privacy_mask, (35, 35), 0)
            
            float_privacy_mask = blurred_privacy_mask.astype(np.float32) / 255.0
            three_channel_mask = cv2.merge([float_privacy_mask, float_privacy_mask, float_privacy_mask]) 
            camera_frame = (camera_frame.astype(np.float32) * three_channel_mask).astype(np.uint8)
        else:
            camera_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    if not is_camera_showing:
        camera_frame = np.zeros((720, 1280, 3), np.uint8) 

    if is_grid_mode_active:
        for y_coordinate in range(0, 720, 40): 
            cv2.line(camera_frame, (0, y_coordinate), (1280, y_coordinate), (50, 50, 50), 1)
        for x_coordinate in range(0, 1280, 40): 
            cv2.line(camera_frame, (x_coordinate, 0), (x_coordinate, 720), (50, 50, 50), 1)

    for color_index in range(len(available_colors)):
        rect_x1, rect_y1, rect_x2, rect_y2 = color_rectangles[color_index]
        cv2.rectangle(camera_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), available_colors[color_index], cv2.FILLED)
        
        if current_draw_color == available_colors[color_index]:
            cv2.rectangle(camera_frame, (rect_x1 - 5, rect_y1 - 5), (rect_x2 + 5, rect_y2 + 5), (255, 255, 255), 4) 
            
    cv2.rectangle(camera_frame, (20, 120), (120, 200), (100, 100, 100), cv2.FILLED)
    cv2.putText(camera_frame, "UNDO", (35, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(camera_frame, "1 FINGER: Draw | 2: Menu/Erase | 3: Circle | 4: Clear", (20, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(camera_frame, "KEYS -> B:Bg | S:Save | G:Grid | M:Measure | P:Privacy | L:Laser | E:Eyedrop", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if display_status_timer > 0:
        cv2.putText(camera_frame, display_status_text, (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, current_draw_color, 5)
        display_status_timer -= 1

    if hand_tracking_results.multi_hand_landmarks:
        cv2.putText(camera_frame, "AI STATUS: TRACKING", (1000, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for hand_landmarks_data in hand_tracking_results.multi_hand_landmarks:
            mediapipe_drawing.draw_landmarks(camera_frame, hand_landmarks_data, mediapipe_hands.HAND_CONNECTIONS)
            
            finger_landmarks = hand_landmarks_data.landmark
            raw_index_x = int(finger_landmarks[8].x * frame_width)
            raw_index_y = int(finger_landmarks[8].y * frame_height)
            thumb_x = int(finger_landmarks[4].x * frame_width)
            thumb_y = int(finger_landmarks[4].y * frame_height)
            
            if smoothed_x == 0 and smoothed_y == 0:
                smoothed_x = raw_index_x
                smoothed_y = raw_index_y
            else:
                smoothed_x = int(smoothed_x + (raw_index_x - smoothed_x) * 0.5)
                smoothed_y = int(smoothed_y + (raw_index_y - smoothed_y) * 0.5)
            
            index_finger_x = smoothed_x
            index_finger_y = smoothed_y 
            
            depth_distance_x = finger_landmarks[0].x * frame_width - finger_landmarks[9].x * frame_width
            depth_distance_y = finger_landmarks[0].y * frame_height - finger_landmarks[9].y * frame_height
            hand_depth_proxy = math.hypot(depth_distance_x, depth_distance_y)
            
            raw_brush_thickness = int(np.interp(hand_depth_proxy, [50, 250], [2, 30]))
            smoothed_thickness = int(smoothed_thickness + (raw_brush_thickness - smoothed_thickness) * 0.2)

            if is_measure_mode_active:
                cv2.line(camera_frame, (thumb_x, thumb_y), (raw_index_x, raw_index_y), (255, 0, 255), 3)
                cv2.circle(camera_frame, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(camera_frame, (raw_index_x, raw_index_y), 10, (255, 0, 255), cv2.FILLED)
                
                caliper_distance_pixels = math.hypot(raw_index_x - thumb_x, raw_index_y - thumb_y)
                estimated_centimeters = caliper_distance_pixels / 25.0
                
                midpoint_x = (thumb_x + raw_index_x) // 2
                midpoint_y = (thumb_y + raw_index_y) // 2 - 20
                cv2.putText(camera_frame, f"{estimated_centimeters:.1f} cm", (midpoint_x, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            wrist_landmark = finger_landmarks[0] 

            is_index_up = is_finger_open(8, 6, finger_landmarks, wrist_landmark)
            is_middle_up = is_finger_open(12, 10, finger_landmarks, wrist_landmark)
            is_ring_up = is_finger_open(16, 14, finger_landmarks, wrist_landmark)
            is_pinky_up = is_finger_open(20, 18, finger_landmarks, wrist_landmark)

            if 20 < index_finger_x < 120 and 120 < index_finger_y < 200:
                undo_hover_timer += 1
                cv2.ellipse(camera_frame, (70, 160), (45, 45), 0, 0, undo_hover_timer * 18, (0, 255, 0), 4) 
                if undo_hover_timer > 20: 
                    if canvas_history_stack:
                        drawing_canvas = canvas_history_stack.pop() 
                    undo_hover_timer = 0
            else:
                undo_hover_timer = 0

            if is_index_up and is_middle_up and is_ring_up and not is_pinky_up:
                circle_radius = int(math.hypot(thumb_x - index_finger_x, thumb_y - index_finger_y)) 
                cv2.circle(camera_frame, (index_finger_x, index_finger_y), circle_radius, current_draw_color, 2) 
                was_in_shape_mode = True
                last_drawn_circle = (index_finger_x, index_finger_y, circle_radius)
                is_user_drawing = False
                previous_x = 0
                previous_y = 0
                clear_board_timer = 0
            else:
                if was_in_shape_mode and last_drawn_circle:
                    canvas_history_stack.append(drawing_canvas.copy())
                    if len(canvas_history_stack) > 5: 
                        canvas_history_stack.pop(0)
                        
                    center_x, center_y, final_radius = last_drawn_circle
                    safe_thickness = max(3, smoothed_thickness)
                    cv2.circle(drawing_canvas, (center_x, center_y), final_radius, current_draw_color, safe_thickness)
                    
                    was_in_shape_mode = False
                    last_drawn_circle = None
                    is_user_drawing = False
                    previous_x = 0
                    previous_y = 0
                    clear_board_timer = 0
                    continue 

                if is_index_up and is_middle_up and is_ring_up and is_pinky_up:
                    clear_board_timer += 1
                    cv2.ellipse(camera_frame, (index_finger_x, index_finger_y), (60, 60), 0, 0, clear_board_timer * 18, (0, 0, 255), 4)
                    
                    if clear_board_timer > 20:
                        if len(np.unique(drawing_canvas)) > 1: 
                            canvas_history_stack.append(drawing_canvas.copy())
                            if len(canvas_history_stack) > 5: 
                                canvas_history_stack.pop(0)
                                
                        drawing_canvas = np.zeros((720, 1280, 3), np.uint8) 
                        display_status_text = "CLEARED!"
                        display_status_timer = 15
                        is_user_drawing = False
                        previous_x = 0
                        previous_y = 0
                        clear_board_timer = 0
                else:
                    clear_board_timer = 0

                if is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
                    if not is_user_drawing:
                        canvas_history_stack.append(drawing_canvas.copy())
                        if len(canvas_history_stack) > 5: 
                            canvas_history_stack.pop(0)
                        is_user_drawing = True
                        
                    half_thickness = int(smoothed_thickness / 2)
                    cv2.circle(camera_frame, (index_finger_x, index_finger_y), half_thickness, current_draw_color, cv2.FILLED)
                    
                    if previous_x == 0 and previous_y == 0: 
                        previous_x = index_finger_x
                        previous_y = index_finger_y
                        
                    cv2.line(drawing_canvas, (previous_x, previous_y), (index_finger_x, index_finger_y), current_draw_color, smoothed_thickness)
                    previous_x = index_finger_x
                    previous_y = index_finger_y

                elif is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
                    is_user_drawing = False
                    previous_x = 0
                    previous_y = 0 
                    
                    if index_finger_y < 130: 
                        for color_index in range(len(color_rectangles)):
                            rect_x1, rect_y1, rect_x2, rect_y2 = color_rectangles[color_index]
                            if rect_x1 <= index_finger_x <= rect_x2:
                                current_draw_color = available_colors[color_index]
                                cv2.circle(camera_frame, (index_finger_x, index_finger_y), 20, (255, 255, 255), cv2.FILLED) 
                    else:
                        half_eraser = int(eraser_brush_thickness / 2)
                        cv2.circle(camera_frame, (index_finger_x, index_finger_y), eraser_brush_thickness, (0, 0, 255), 2)        
                        cv2.circle(camera_frame, (index_finger_x, index_finger_y), half_eraser, (0, 0, 255), 1) 
                        cv2.circle(drawing_canvas, (index_finger_x, index_finger_y), eraser_brush_thickness, (0, 0, 0), cv2.FILLED) 
                
                elif not (is_index_up and is_middle_up and is_ring_up and is_pinky_up):
                    is_user_drawing = False
                    previous_x = 0
                    previous_y = 0
    else:
        cv2.putText(camera_frame, "AI STATUS: SEARCHING...", (1000, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        smoothed_x = 0
        smoothed_y = 0 
        is_user_drawing = False
        clear_board_timer = 0

    if is_laser_mode_active:
        drawing_canvas = cv2.addWeighted(drawing_canvas, 0.92, drawing_canvas, 0, 0)

    camera_frame = cv2.addWeighted(camera_frame, 1, drawing_canvas, 1, 0)
    cv2.imshow("Virtual Whiteboard", camera_frame)
    
    keyboard_input = cv2.waitKey(1) & 0xFF
    
    if keyboard_input == ord('q'): 
        break
    elif keyboard_input == ord('c'): 
        drawing_canvas = np.zeros((720, 1280, 3), np.uint8)
    elif keyboard_input == ord('b'): 
        is_camera_showing = not is_camera_showing 
    elif keyboard_input == ord('g'): 
        is_grid_mode_active = not is_grid_mode_active 
    elif keyboard_input == ord('m'): 
        is_measure_mode_active = not is_measure_mode_active 
    elif keyboard_input == ord('p'): 
        is_privacy_mode_active = not is_privacy_mode_active 
    elif keyboard_input == ord('l'): 
        is_laser_mode_active = not is_laser_mode_active
        if is_laser_mode_active:
            display_status_text = "LASER ON!"
        else:
            display_status_text = "LASER OFF!"
        display_status_timer = 20
    elif keyboard_input == ord('e') and hand_tracking_results.multi_hand_landmarks:
        safe_index_x = max(0, min(index_finger_x, 1279))
        safe_index_y = max(0, min(index_finger_y, 719))
        
        blue_val, green_val, red_val = clean_camera_feed[safe_index_y, safe_index_x]
        current_draw_color = (int(blue_val), int(green_val), int(red_val)) 
        
        display_status_text = "COLOR COPIED!"
        display_status_timer = 20
    elif keyboard_input == ord('s'):
        current_timestamp = int(time.time()) 
        cv2.imwrite(f"StudyNotes_{current_timestamp}.png", drawing_canvas)
        display_status_text = "SAVED TO FOLDER!"
        display_status_timer = 20

video_capture.release()
cv2.destroyAllWindows()