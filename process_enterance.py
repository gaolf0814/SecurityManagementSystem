import time
from multiprocessing import Process, Value
from main import gui_main
from handler.monitor import is_alive
from configs.config import check_detection_process_interval, reboot_time_steps


def subprocess_run(detection_flag: Value) -> Process:
    p = Process(target=gui_main, args=(detection_flag,))
    p.start()
    return p


def main():
    detection_flag = Value('i', 0)  # variable(integer) with shared memory between multi processes
    p = subprocess_run(detection_flag)
    pre_time = time.time()

    while True:
        now_time = time.time()
        if now_time - pre_time > reboot_time_steps * 3600:
            print('The detection process is <dead>!')
            p.terminate()  # kill the subprocess
            time.sleep(1)
            print('reboot')
            detection_flag = Value('i', 0)
            p = subprocess_run(detection_flag)
            pre_time = now_time

        time.sleep(check_detection_process_interval)
        if is_alive(detection_flag):
            print('The detection process is <alive>')
        else:
            print('The detection process is <dead>!')
            p.terminate()  # kill the subprocess
            time.sleep(1)
            print('reboot')
            detection_flag = Value('i', 0)
            p = subprocess_run(detection_flag)


if __name__ == '__main__':
    main()
