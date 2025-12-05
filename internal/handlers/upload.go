package handlers

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"jaybird-backend/internal/models"
	"jaybird-backend/internal/services"

	"github.com/gin-gonic/gin"
)

type UploadHandler struct {
	cloudinaryService *services.CloudinaryService
	inferenceService  *services.InferenceService
	tempDir           string
}

func NewUploadHandler(cloudinaryService *services.CloudinaryService, inferenceService *services.InferenceService, tempDir string) *UploadHandler {
	return &UploadHandler{
		cloudinaryService: cloudinaryService,
		inferenceService:  inferenceService,
		tempDir:           tempDir,
	}
}

func (h *UploadHandler) Handle(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(400, models.ErrorResponse{Error: "No file provided"})
		return
	}

	title := c.PostForm("title")
	tagsStr := c.PostForm("tags")
	tagsArray := c.PostFormArray("tags")

	// Combine tags from both sources
	var userTags []string
	if len(tagsArray) > 0 {
		userTags = tagsArray
	} else if tagsStr != "" {
		userTags = strings.Split(tagsStr, ",")
	}

	// Validate file type
	contentType := file.Header.Get("Content-Type")
	validImageTypes := []string{"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
	validVideoTypes := []string{"video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"}

	isValidImage := contains(validImageTypes, contentType)
	isValidVideo := contains(validVideoTypes, contentType)

	if !isValidImage && !isValidVideo {
		c.JSON(400, models.ErrorResponse{Error: "Invalid file type. Only images and videos are allowed."})
		return
	}

	resourceType := "image"
	if isValidVideo {
		resourceType = "video"
	}

	src, err := file.Open()
	if err != nil {
		c.JSON(500, models.ErrorResponse{Error: "Failed to open uploaded file"})
		return
	}
	defer src.Close()

	// Determine if we need temp file (for large files > 50MB)
	fileSize := file.Size
	LARGE_FILE_THRESHOLD := int64(50 * 1024 * 1024) // 50MB
	isLargeFile := fileSize > LARGE_FILE_THRESHOLD

	var fileData io.Reader
	var tempFilePath string
	var buffer []byte

	if isLargeFile {
		// Save to temp file for large files
		ext := filepath.Ext(file.Filename)
		if ext == "" {
			// Determine extension from content type
			exts, _ := mime.ExtensionsByType(contentType)
			if len(exts) > 0 {
				ext = exts[0]
			} else {
				ext = ".tmp"
			}
		}

		tempFile, err := os.CreateTemp(h.tempDir, fmt.Sprintf("upload-*%s", ext))
		if err != nil {
			c.JSON(500, models.ErrorResponse{Error: "Failed to create temporary file"})
			return
		}
		tempFilePath = tempFile.Name()
		defer os.Remove(tempFilePath)

		_, err = io.Copy(tempFile, src)
		tempFile.Close()
		if err != nil {
			c.JSON(500, models.ErrorResponse{Error: "Failed to save temporary file"})
			return
		}

		tempFileReader, err := os.Open(tempFilePath)
		if err != nil {
			c.JSON(500, models.ErrorResponse{Error: "Failed to read temporary file"})
			return
		}
		defer tempFileReader.Close()
		fileData = tempFileReader
	} else {
		// Read into memory for small files
		buffer = make([]byte, fileSize)
		_, err = src.Read(buffer)
		if err != nil && err != io.EOF {
			c.JSON(500, models.ErrorResponse{Error: "Failed to read file data"})
			return
		}
		fileData = bytes.NewReader(buffer)
	}

	var mlTags []string
	var allTags []string

	// Prepare image data for inference
	if isValidImage {
		var imageData []byte
		if isLargeFile {
			imageData, err = os.ReadFile(tempFilePath)
			if err != nil {
				// Continue without ML tags if we can't read the file
				mlTags = []string{}
			}
		} else {
			imageData = buffer
		}

		if len(imageData) > 0 {
			inferenceResult, err := h.inferenceService.GetMLTags(imageData)
			if err == nil && inferenceResult != nil {
				mlTags = inferenceResult.Tags
				if len(mlTags) > 0 {
					log.Printf("[UploadHandler] Added %d ML tags to upload", len(mlTags))
				}
			} else if err != nil {
				log.Printf("[UploadHandler] Inference service failed: %v (proceeding with user tags only)", err)
			}
		}
	}

	// Combine user tags and ML tags
	allTags = combineTags(userTags, mlTags)

	// Upload to Cloudinary
	ctx := context.Background()
	uploadResult, err := h.cloudinaryService.UploadFile(ctx, fileData, allTags, title, resourceType)
	if err != nil {
		c.JSON(500, models.ErrorResponse{Error: fmt.Sprintf("Failed to upload to Cloudinary: %v", err)})
		return
	}

	c.JSON(200, models.UploadResponse{
		Success:  true,
		URL:      uploadResult.SecureURL,
		PublicID: uploadResult.PublicID,
		Tags:     allTags,
	})
}

func combineTags(userTags []string, mlTags []string) []string {
	normalizedUserTags := normalizeTags(userTags)
	normalizedMlTags := normalizeTags(mlTags)
	combined := append(normalizedUserTags, normalizedMlTags...)
	return combined
}

func normalizeTags(tags []string) []string {
	seen := make(map[string]bool)
	var res []string

	for _, tag := range tags {
		normalized := strings.ToLower(strings.TrimSpace(tag))
		if normalized != "" && !seen[normalized] {
			seen[normalized] = true
			res = append(res, normalized)
		}
	}
	return res
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
