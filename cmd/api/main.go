package main

import (
	"jaybird-backend/internal/config"
	"jaybird-backend/internal/services"
	"jaybird-backend/routes"
	"log"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	// 1. Load configuration using config.LoadConfig()
	config, err := config.LoadConfig()
	if err != nil {
		log.Fatal("Failed to load config:", err)
	}

	// 2. Initialize Cloudinary service using services.NewCloudinaryService()
	cloudinaryService, err := services.NewCloudinaryService(
		config.CloudinaryCloudName,
		config.CloudinaryAPIKey,
		config.CloudinaryAPISecret,
	)
	if err != nil {
		log.Fatal("Failed to initialize Cloudinary service:", err)
	}

	// 3. Initialize inference service using services.NewInferenceService()
	inferenceService := services.NewInferenceService(config.PythonInferenceURL)

	// 4. Setup Gin router with gin.Default()
	router := gin.Default()

	// Add CORS middleware
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{config.CORSAllowedURL},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// 5. Setup routes using routes.SetupRoutes()
	routes.SetupRoutes(router, cloudinaryService, inferenceService, config.TempDir)

	log.Printf("Server starting on port %s", config.Port)
	if err := router.Run(":" + config.Port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
