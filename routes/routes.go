package routes

import (
	"jaybird-backend/internal/handlers"
	"jaybird-backend/internal/models"
	"jaybird-backend/internal/services"

	"github.com/gin-gonic/gin"
)

// SetupRoutes configures all routes for the application
func SetupRoutes(router *gin.Engine, cloudinaryService *services.CloudinaryService, inferenceService *services.InferenceService, tempDir string) {
	router.GET("/api/v1/health", healthHandler)

	uploadHandler := handlers.NewUploadHandler(cloudinaryService, inferenceService, tempDir)
	router.POST("/api/v1/upload", uploadHandler.Handle)
}

func healthHandler(c *gin.Context) {
	c.JSON(200, models.UploadResponse{
		Success: true,
	})
}
