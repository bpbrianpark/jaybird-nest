package config

import (
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	Port                string
	CloudinaryCloudName string
	CloudinaryAPIKey    string
	CloudinaryAPISecret string
	CORSAllowedURL      string
	PythonInferenceURL  string
	TempDir             string
}

func LoadConfig() (*Config, error) {
	godotenv.Load()

	config := &Config{
		Port:                os.Getenv("PORT"),
		CloudinaryCloudName: os.Getenv("CLOUDINARY_CLOUD_NAME"),
		CloudinaryAPIKey:    os.Getenv("CLOUDINARY_API_KEY"),
		CloudinaryAPISecret: os.Getenv("CLOUDINARY_API_SECRET"),
		CORSAllowedURL:      os.Getenv("CORS_ALLOWED_URL"),
		PythonInferenceURL:  getEnvOrDefault("PYTHON_INFERENCE_URL", "http://localhost:8000"),
		TempDir:             getEnvOrDefault("TEMP_DIR", os.TempDir()),
	}

	return config, nil
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
