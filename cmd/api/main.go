package main

import (
	"github.com/gin-gonic/gin"
)

func main() {
	// TODO: Implement Go API entry point
	// 1. Load configuration using config.LoadConfig()
	// 2. Initialize Cloudinary service using services.NewCloudinaryService()
	// 3. Initialize inference service using services.NewInferenceService()
	// 4. Setup Gin router with gin.Default()
	// 5. Setup routes using routes.SetupRoutes()
	// 6. Start server with router.Run(":" + config.Port)

	router := gin.Default()
	router.POST("/upload", healthHandler)
	router.GET("/api/v1/health", healthHandler)
	router.Run("localhost:8080")
}

func healthHandler(c *gin.Context) {
	c.JSON(200, gin.H{"status": "ok"})

}
