//#include <SoftwareSerial.h>
int led=13;
int data=5;

int user_input;

# define led_pin 8
# define LIGHT_PIN 6
# define HIGH_LIGHT_PIN 7

# define MOTION_SENSOR 2

int LIGHT_SENSOR_PIN = A0;
int light_value = 0;
int user =3;

bool dim_light = 1;
bool motion_detected = 0;


//SoftwareSerial BTSerial(0, 1);
void setup() {
  // put your setup code here, to run once:



  Serial.begin(9600);
  //BTSerial.begin(9600);
  pinMode(13,OUTPUT);
  pinMode(MOTION_SENSOR, INPUT);
  pinMode(LIGHT_PIN, OUTPUT); // low on
  pinMode(HIGH_LIGHT_PIN, OUTPUT); // high on

}

void loop() {
 
 delay(200);
  int sensorValue = analogRead(A0);

 // light_value = analogRead(LIGHT_SENSOR_PIN);
if(user==3){
 if(sensorValue > 300) {
    
    dim_light = 1;
    Serial.println("Brigth Light");
     Serial.println("Light sensor value:");
     Serial.println(sensorValue);
  }
  else { 
    dim_light = 0;
    Serial.println("Dim Light");
    Serial.println("Light sensor value:");
    Serial.println(sensorValue);
  }

    
  if(digitalRead(MOTION_SENSOR) == 1 ) {
    delay(1000);
    if(digitalRead(MOTION_SENSOR) == 1){
      motion_detected = 1;
      Serial.println("Motion detected");
    }
  }
  if(digitalRead(MOTION_SENSOR) == 0) {
    delay(1000);
    if(digitalRead(MOTION_SENSOR) == 0){
      motion_detected = 0;
      Serial.println("NO Motion detected");
    }
  }
      // Send data over Bluetooth
  //BTSerial.write(sensorValue);

  if(motion_detected == 1){
    digitalWrite(LIGHT_PIN,HIGH);
    digitalWrite(HIGH_LIGHT_PIN,LOW);
    //digitalWrite(HIGH_LIGHT_PIN, LOW);
    Serial.println(" ON");
    if(dim_light == 0) {
      digitalWrite(LIGHT_PIN,HIGH);
      digitalWrite(HIGH_LIGHT_PIN,LOW);
      Serial.println("Night, Using Full Light");
    }
    else{
      digitalWrite(LIGHT_PIN,HIGH);
      digitalWrite(HIGH_LIGHT_PIN,HIGH);
      Serial.println("SunLight, Using Dimmed Light");
    } 
    delay(1000);

  }
  else {
    digitalWrite(LIGHT_PIN,LOW);
    digitalWrite(HIGH_LIGHT_PIN,HIGH);
    Serial.println("OFF");
    delay(1000);

  }
}

while(Serial.available()>0){
   // Serial.println("Started manual mode");
    user=Serial.read();
    Serial.println(user);
    switch (user) {
      case 48:
        digitalWrite(LIGHT_PIN,LOW);
        digitalWrite(HIGH_LIGHT_PIN,HIGH);
        break;
      case 49:
        digitalWrite(LIGHT_PIN,HIGH);
        digitalWrite(HIGH_LIGHT_PIN,HIGH);
        break;
      case 50:
        digitalWrite(LIGHT_PIN,HIGH);
        digitalWrite(HIGH_LIGHT_PIN,LOW);
        break;
      case 51:
        user=3;
       // Serial.println("Started automated mode");
      
      }
    
}

 
  
  
 


  

  
}

