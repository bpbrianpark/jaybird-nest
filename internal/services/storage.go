package services

import (
	"context"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/cloudinary/cloudinary-go/v2"
	"github.com/cloudinary/cloudinary-go/v2/api"
	"github.com/cloudinary/cloudinary-go/v2/api/uploader"
)

type CloudinaryService struct {
	cld *cloudinary.Cloudinary
}

type UploadResult struct {
	URL       string
	PublicID  string
	SecureURL string
}

func NewCloudinaryService(cloudName, apiKey, apiSecret string) (*CloudinaryService, error) {
	cld, err := cloudinary.NewFromParams(cloudName, apiKey, apiSecret)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Cloudinary: %w", err)
	}
	return &CloudinaryService{cld: cld}, nil
}

func (s *CloudinaryService) UploadFile(ctx context.Context, fileData io.Reader, tags []string, title string, resourceType string) (*UploadResult, error) {
	normalizedTags := normalizeTags(tags)

	params := uploader.UploadParams{
		Folder: "gallery",
		Tags:   normalizedTags,
	}

	if title != "" {
		params.Context = api.CldAPIMap{
			"title": strings.TrimSpace(title),
		}
	}

	if resourceType == "video" {
		params.Eager = "mp4"
		eagerAsync := false
		params.EagerAsync = &eagerAsync
	}

	result, err := s.cld.Upload.Upload(ctx, fileData, params)
	if err != nil {
		return nil, fmt.Errorf("failed to upload to Cloudinary: %w", err)
	}

	return &UploadResult{
		URL:       result.URL,
		PublicID:  result.PublicID,
		SecureURL: result.SecureURL,
	}, nil
}

func normalizeTags(input []string) []string {
	if len(input) == 0 {
		return []string{}
	}

	seen := make(map[string]bool)
	var normalized []string

	// Regular expression to remove special characters (keep only word chars, spaces, and hyphens)
	re := regexp.MustCompile(`[^\w\s-]`)

	for _, tag := range input {
		parts := strings.Split(tag, ",")
		for _, part := range parts {
			trimmed := strings.TrimSpace(part)
			if trimmed == "" {
				continue
			}

			lower := strings.ToLower(trimmed)

			cleaned := re.ReplaceAllString(lower, "")

			slugified := strings.ReplaceAll(cleaned, " ", "-")

			slugified = regexp.MustCompile(`-+`).ReplaceAllString(slugified, "-")

			slugified = strings.Trim(slugified, "-")

			if slugified != "" && !seen[slugified] {
				seen[slugified] = true
				normalized = append(normalized, slugified)
			}
		}
	}

	return normalized
}
