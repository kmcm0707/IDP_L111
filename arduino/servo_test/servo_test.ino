// ARDUINO SERVO CODE

//N.B. If we need to turn off the servos, we can servo.attach at the start of each function, and servo.detach at the end of each function

#include <Servo.h>
//#include <Adafruit_MotorShield.h>

//Adafruit_MotorShield AFMS = Adafruit_MotorShield();

int servo_vertical_pin = 9;
int servo_horizontal_pin = 10;

int vertical_angle_high = 125; //120?
int vertical_angle_low = 25;
int horizontal_angle_high = 90; //95?
int horizontal_angle_low = 0;

Servo servo_vertical;
Servo servo_horizontal;

int pos = 0;

void setup() {
  //if (!AFMS.begin()) {
    //while (1);
  //}

  servo_horizontal.attach(servo_horizontal_pin);
  servo_vertical.attach(servo_vertical_pin);

}

void loop() {
  Serial.write(servo_horizontal.read());
  Serial.write('\n');

  close_claw();
  // open_claw();
  raise_claw();
  delay(5000);
  lower_claw();
  open_claw();
  delay(5000);
}

void lower_claw(){
  servo_vertical.attach(servo_vertical_pin);
  // From arduino 'servo' example:
  for (pos = vertical_angle_high; pos >= vertical_angle_low; pos -= 1) {
    // in steps of 1 degree
    servo_vertical.write(pos);              // tell servo to go to position in variable 'pos'
    delay(15);                                // waits 15 ms for the servo to reach the position
  }
  servo_vertical.detach();
}

void raise_claw(){
  servo_vertical.attach(servo_vertical_pin);
  // From arduino 'servo' example:
  for (pos = vertical_angle_low; pos <= vertical_angle_high; pos += 1) {
    // in steps of 1 degree
    servo_vertical.write(pos);              // tell servo to go to position in variable 'pos'
    delay(15);                                // waits 15 ms for the servo to reach the position
  
  }
  servo_vertical.detach();
}

void close_claw(){
  // close the lower section of the claw (grab the block)
  // From arduino 'servo' example:
  for (pos = horizontal_angle_low; pos <= horizontal_angle_high; pos += 1) {
    // in steps of 1 degree
    servo_horizontal.write(pos);              // tell servo to go to position in variable 'pos'
    delay(15);                                // waits 15 ms for the servo to reach the position
  
  }
}

void open_claw(){
  // open the lower section of the claw (release the block)
  // From arduino 'servo' example:
  for (pos = horizontal_angle_high; pos >= horizontal_angle_low; pos -= 1) {
    // in steps of 1 degree
    servo_horizontal.write(pos);              // tell servo to go to position in variable 'pos'
    delay(15);                                // waits 15 ms for the servo to reach the position
  }
}
