package models

// UploadResponse represents the success response for file uploads
type UploadResponse struct {
	Success  bool     `json:"success"`
	URL      string   `json:"url"`
	PublicID string   `json:"public_id"`
	Tags     []string `json:"tags,omitempty"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

type InferenceResult struct {
	Tags []string `json:"tags"`
}

type InferenceRequest struct {
	ImageData string `json:"image_data"`
}
