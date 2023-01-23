#include <Adafruit_MotorShield.h>

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1);
Adafruit_DCMotor *m2 = AFMS.getMotor(2);

int left_line_follower = A0;
int right_line_follower = A1;
int left_sensorValue = 0;
int right_sensorValue = 0;
int linefollower_trigger = 600;
void setup() {
  Serial.begin(9600);

  //line folllower
  // put your setup code here, to run once:
  
  //motor setup
  
  if (!AFMS.begin()) {         // create with the default frequency 1.6KHz
  // if (!AFMS.begin(1000)) {  // OR with a different frequency, say 1KHz
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  Serial.println("Motor Shield found.");

  // Set the speed to start, from 0 (off) to 255 (max speed)
  m1->setSpeed(150);
  m2->setSpeed(150);
  m1->run(FORWARD);
  // turn on motor
  m1->run(RELEASE);
  
}

void loop() {
  uint8_t i;
  left_sensorValue = analogRead(left_line_follower);
  Serial.println(left_sensorValue);
  m1->run(FORWARD);
  for (i=0; i<255; i++) {
    m1->setSpeed(i);
    delay(10);
  }
  for (i=255; i!=0; i--) {
    m1->setSpeed(i);
    delay(10);
  }

  Serial.print("tock");

  m1->run(BACKWARD);
  for (i=0; i<255; i++) {
    m1->setSpeed(i);
    delay(10);
  }
  for (i=255; i!=0; i--) {
    m1->setSpeed(i);
    delay(10);
  }

  Serial.print("tech");
  m1->run(RELEASE);
  delay(10);
}
void line_following(){
  if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // Move Forward
  {
  m1->run(FORWARD);
  m2->run(FORWARD);
  }
  
  if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // Turn right
  {
  m1->run(BACKWARD);
  m2->run(FORWARD);
  }
  
  if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // turn left
  {
  m1->run(FORWARD);
  m2->run(BACKWARD);
  }
  
  if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // stop !(digitalRead(LS)) && !(digitalRead(RS))
  {
  
  }
}
