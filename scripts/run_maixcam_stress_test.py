"""
Upload and run stress test on MaixCAM device.
"""
import os
import paramiko
import time

MAIXCAM_HOST = os.getenv("MAIXCAM_HOST", "10.154.35.1")
MAIXCAM_PORT = int(os.getenv("MAIXCAM_PORT", "22"))
MAIXCAM_USER = os.getenv("MAIXCAM_USER", "root")
MAIXCAM_PASS = os.getenv("MAIXCAM_PASS", "root")

LOCAL_SCRIPT = "MaixCAM_App/stress_test_maixcam.py"
REMOTE_SCRIPT = "/root/stress_test_maixcam.py"

def main():
    print(f"Connecting to MaixCAM {MAIXCAM_HOST}:{MAIXCAM_PORT}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            MAIXCAM_HOST,
            port=MAIXCAM_PORT,
            username=MAIXCAM_USER,
            password=MAIXCAM_PASS,
            timeout=10,
        )
    except Exception as e:
        print(f"[ERROR] Cannot connect: {e}")
        return

    sftp = client.open_sftp()

    # Upload stress test script
    print(f"Uploading stress test script...")
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
    print(f"  -> {REMOTE_SCRIPT}")

    sftp.close()

    # Run stress test
    print("\n" + "="*60)
    print("  RUNNING STRESS TEST ON MAIXCAM")
    print("="*60 + "\n")

    stdin, stdout, stderr = client.exec_command(
        f"cd /root && python stress_test_maixcam.py 100",
        timeout=300
    )

    # Stream output
    for line in iter(lambda: stdout.readline(), ''):
        if line:
            print(line, end='')

    stderr_output = stderr.read().decode()
    if stderr_output:
        print(f"\n[STDERR]\n{stderr_output}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
