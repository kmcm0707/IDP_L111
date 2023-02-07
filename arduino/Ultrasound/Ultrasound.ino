#include <HCSR04.h>
const int trigPin = 6;
const int echoPin = 7;
// defines variables
long duration;
int distance;
UltraSonicDistanceSensor distancesensor(trigPin, echoPin);

void setup() {
  Serial.begin(9600); // Starts the serial communication
}

void loop() {
  Serial.println(distancesensor.measureDistanceCm());
  delay(100);
}
