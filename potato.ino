#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C  // Use 0x3D if OLED doesn't work
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Structure to hold classification data
struct ClassificationData {
  String label;
  float confidence;
  bool valid;
  unsigned long lastUpdate;
};

ClassificationData currentData = {"", 0.0, false, 0};

void setup() {
  Serial.begin(9600);  // Initialize serial for debugging
  while (!Serial);     // Wait for serial monitor (optional)

  // Initialize OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("OLED allocation failed!"));
    while (1);  // Freeze if OLED fails
  }

  // Show startup screen
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("Potato Disease");
  display.println("Classifier");
  display.println("Waiting for data...");
  display.display();
  delay(2000);
}

void processSerialData() {
  if (Serial.available()) {
    String jsonStr = Serial.readStringUntil('\n');
    jsonStr.trim();

    // Debug output (view in Serial Monitor)
    Serial.print("Received: ");
    Serial.println(jsonStr);

    // Parse JSON manually (lightweight for Arduino)
    int labelStart = jsonStr.indexOf("\"label\":\"") + 8;
    int labelEnd = jsonStr.indexOf("\"", labelStart);
    int confStart = jsonStr.indexOf("\"confidence\":") + 12;
    int confEnd = jsonStr.indexOf(",", confStart);
    if (confEnd == -1) confEnd = jsonStr.indexOf("}", confStart);

    if (labelStart > 0 && labelEnd > 0 && confStart > 0 && confEnd > 0) {
      currentData.label = jsonStr.substring(labelStart, labelEnd);
      currentData.confidence = jsonStr.substring(confStart, confEnd).toFloat();
      currentData.lastUpdate = millis();
      currentData.valid = true;

      // Debug output
      Serial.print("Label: ");
      Serial.print(currentData.label);
      Serial.print(", Confidence: ");
      Serial.println(currentData.confidence);
    }
  }
}

void updateDisplay() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  
  display.println("Potato Disease");
  display.println("--------------");

  if (currentData.valid) {
    // Display disease label (with wrapping)
    display.print("Disease: ");
    if (currentData.label.length() > 12) {
      display.println(currentData.label.substring(0, 12));
      display.println(currentData.label.substring(12));
    } else {
      display.println(currentData.label);
    }

    // Display confidence as percentage
    display.print("Confidence: ");
    display.print(currentData.confidence * 100, 1);
    display.println("%");

    // Show data freshness
    unsigned long secondsAgo = (millis() - currentData.lastUpdate) / 1000;
    display.print("Updated: ");
    display.print(secondsAgo);
    display.println("s ago");
  } else {
    display.println("No data received");
    display.println("Check connection");
  }

  display.display();
}

void loop() {
  processSerialData();  // Check for new serial data
  updateDisplay();      // Refresh OLED
  delay(500);           // Update twice per second
}
