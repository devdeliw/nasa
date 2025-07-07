import time, keyboard

def keep_awake(interval_minutes: float = 2.0, duration_minutes: float = 0.0):
    interval_secs = interval_minutes * 60
    end_time = time.time() + duration_minutes * 60 if duration_minutes > 0 else None

    print(f"keep-awake started: every {interval_minutes} min", end='')
    print(f", for {duration_minutes} min." if duration_minutes > 0 else ", indefinitely.")
    
    press_ct = 0
    try:
        while end_time is None or time.time() < end_time:
            keyboard.press_and_release('f15')
            press_ct += 1
            time.sleep(interval_secs)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("Keep-awake session ended.")
        print(f"F15 pressed {press_ct} times.")


keep_awake(2.0, 0.0)   
