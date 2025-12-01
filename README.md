# Behavioral-Authentication-on-M5Stack-Tab5
Behavioral authentication system using typing dynamics on M5Stack Tab5.

Project Overview

This project implements a behavioral biometric authentication system on the M5Stack Tab5 using typing dynamics.
Instead of validating a user solely based on a PIN code, the device analyzes how the PIN is typed, using timing features that form a unique behavioral signature.

A lightweight neural network (ANN) runs fully on-device, ensuring:

Privacy by design (no cloud, no data upload)

Ultra-low latency (inference < 1 ms)

Offline operation

Real-time learning and evaluation

Installation & Deployment
Requirements
M5Stack Tab5
M5Unified library for Arduino
Arduino IDE or PlatformIO

Installation Steps:

1 Clone this repository.

2 Open the .ino file in the Arduino IDE.

3 Install the required dependencies (M5Unified).

4 Connect the Tab5 via USB-C.

5 Flash the firmware onto the device.

The system boots in TRAIN mode by default. Usage Instructions
TRAIN Mode

In this mode, the system collects typing samples and updates the embedded neural network:

Select the user by tapping the top bar.

Enter multiple samples of the 4-digit code.

The model adapts progressively to the user’s typing behavior.

EVAL Mode

The evaluation mode displays internal model output for transparency:

Predicted user (User0 / User1).

Probability bars for both users.

Running accuracy based on evaluation attempts.

This mode is useful for assessing model quality after training.

TEST Mode

This mode performs the actual authentication decision:

The system makes a final classification.

A confidence bar indicates the reliability of the decision.

Displays the result as “Access Granted” or “Access Denied.”

User selection is locked in this mode for security.

Evaluation Summary

The embedded model was evaluated using both simulated and real data.

Accuracy with simulated data: 85–92%

Accuracy with real user data: 70–85%

Inference speed: below 1 ms

Training update time: below 3 ms

These results confirm that the device is capable of performing real-time behavioral authentication on-device.

Limitations

Requires a sufficient number of training samples per user.

Typing behavior may vary depending on environment and user fatigue.

Current version supports only two users.

Known Issues / Block Points

Touchscreen hitboxes may require manual calibration.

If TRAIN mode is over-used, the ANN can drift away from stable behavior.

The system does not yet detect unknown or impostor users.

Scaling the system to more than two users requires architectural modifications.

Future Improvements

The following enhancements are recommended for future development:

Multi-user support using softmax output or clustering-based classification.

Adaptive learning with decreasing learning rate and stability checks.

Integration of anomaly detection for unknown users.

Vibration or sound feedback to improve user experience.

Stronger machine learning models such as Random Forest, LSTM, or CNN.

Secure local storage of ANN weights using encryption.

License

This project is intended for academic and educational purposes.
You may modify, reuse, or extend the source code for personal or research use.

Author

Behavioral Authentication Project
M5Stack Tab5 – Edge Computing Coursework
