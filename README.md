<div style="display: flex; gap: 10px; flex-wrap: 
nowrap;">
  <img src="/public/jaybirdnest.png" 
alt="Jason Park Photography Logo">
</div>

# Jaybird Nest - Cloudinary Image/Video Uploader

A Go-based REST API for handling file uploads with Cloudinary integration and ML tag inference. Built with Gin framework.

# Why?
I used to have my Next.JS application (https://github.com/bpbrianpark/jaybird/) hooked up to the Cloudinary API directly, but I found that it was quite slow when uploading large amounts of large files.

Goroutines enable multiple file uploads simultaneously, and will be much faster from the user perspective. 

This project was created as a way for me to practice Go and help upload the files without being blocked by the tagging system (which uses the Python ML inference service)

- 📤 **File Upload**: Accept image and video uploads via multipart form data
- ☁️ **Cloudinary Integration**: Automatic upload to Cloudinary with metadata
- 🏷️ **Tag Management**: Combine user-provided tags with ML-generated tags
- 🧠 **ML Tag Inference**: Integrate with Python ML service for automatic tag generation (TODO)
- 📦 **Smart File Handling**: In-memory buffers for small files, temp files for large files (>50MB)
- ✅ **File Validation**: Validates file types (images: JPEG, PNG, GIF, WebP; videos: MP4, MPEG, QuickTime, etc.)
- 🔄 **Graceful Degradation**: Continues upload even if ML service is unavailable

## Prerequisites

- Go 1.23.0 or later
- Cloudinary account (free tier available)

## Installation

1. **Install dependencies**
   ```bash
   go mod download
   ```

2. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   PORT=8080
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   CORS_ALLOWED_URL=your_accepted_url
   TEMP_DIR=/tmp
   ```

   **Getting Cloudinary Credentials:**
   - Sign up at [cloudinary.com](https://cloudinary.com)
   - Go to Dashboard → Settings
   - Copy your Cloud Name, API Key, and API Secret

## Running the Server

```bash
go mod tidy
go run cmd/api/main.go 
```

The server will start on the port specified in your `.env` file.

## API Endpoints

### `GET /api/v1/health`

Health check endpoint.

**Response:**
```json
{
  "success": true
}
```

### `POST /api/v1/upload`

Upload a file (image or video) with optional title and tags.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Fields:
  - `file` (required): The image or video file
  - `title` (optional): Title for the upload
  - `tags` (optional): Comma-separated tags or array of tags

**Success Response (200):**
```json
{
  "success": true,
  "url": "https://res.cloudinary.com/.../image.jpg",
  "public_id": "gallery/abc123",
  "tags": ["nature", "outdoor", "ml-tag-1"]
}
```

**Error Responses:**

- `400 Bad Request`: Missing file or invalid file type
  ```json
  {
    "error": "No file provided"
  }
  ```

- `500 Internal Server Error`: Cloudinary upload failure or server error
  ```json
  {
    "error": "Failed to upload to Cloudinary: ..."
  }
  ```

## File Handling

The service intelligently handles files based on size:

- **Small files (< 50MB)**: Read into memory buffer for faster processing
- **Large files (≥ 50MB)**: Saved to temporary file to avoid memory issues

Temporary files are automatically cleaned up after processing.

## ML Tag Inference (TODO)

## CORS Configuration

CORS is configured to allow requests from `CORS_ALLOWED_URL` in `.env`. To modify allowed origins, update the environment variable.

## License

See [LICENSE](LICENSE) file for details.
