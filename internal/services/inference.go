package services

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"jaybird-backend/internal/models"
)

type InferenceService struct {
	BaseURL    string
	HTTPClient *http.Client
}

func NewInferenceService(baseURL string) *InferenceService {
	return &InferenceService{BaseURL: baseURL, HTTPClient: &http.Client{Timeout: 10 * time.Second}}
}

func (s *InferenceService) GetMLTags(fileData []byte) (*models.InferenceResult, error) {
	imageBase64 := base64.StdEncoding.EncodeToString(fileData)

	requestBody := models.InferenceRequest{
		ImageData: imageBase64,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		log.Printf("[InferenceService] Failed to marshal request: %v", err)
		return &models.InferenceResult{Tags: []string{}}, err
	}

	// Make HTTP POST request to Inference Service
	url := fmt.Sprintf("%s/infer", s.BaseURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("[InferenceService] Failed to create request: %v", err)
		return &models.InferenceResult{Tags: []string{}}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.HTTPClient.Do(req)
	// Service is unavailable - log and continue gracefully
	if err != nil {
		log.Printf("[InferenceService] Failed to call inference service at %s: %v (continuing without ML tags)", url, err)
		return &models.InferenceResult{Tags: []string{}}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("[InferenceService] Inference service returned non-200 status: %s (continuing without ML tags)", resp.Status)
		return &models.InferenceResult{Tags: []string{}}, fmt.Errorf("inference service returned status: %s", resp.Status)
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[InferenceService] Failed to read response body: %v (continuing without ML tags)", err)
		return &models.InferenceResult{Tags: []string{}}, nil
	}

	// Parse JSON response
	var result models.InferenceResult
	if err := json.Unmarshal(body, &result); err != nil {
		log.Printf("[InferenceService] Failed to parse JSON response: %v (continuing without ML tags)", err)
		return &models.InferenceResult{Tags: []string{}}, nil
	}

	log.Printf("[InferenceService] Successfully retrieved %d ML tags", len(result.Tags))
	return &result, nil
}
