int left_line_follower = A0;
int right_line_follower = A1;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  Serial.println(analogRead(left_line_follower));
  Serial.println(analogRead(right_line_follower));
  delay(100); 
}
