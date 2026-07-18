import express from 'express';
import multer from 'multer';
import { businesses } from '../db';
import { uploadFile, validatePhotoFile } from '../services/fileStorage';

const router = express.Router({ mergeParams: true });
const upload = multer({ storage: multer.memoryStorage() });

// POST /api/businesses/:businessId/photo — Upload owner photo
router.post('/', upload.single('photo'), async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;

    if (!req.file) {
      return res.status(400).json({ error: 'No file provided' });
    }

    const validation = await validatePhotoFile({
      mimetype: req.file.mimetype,
      size: req.file.size,
    });
    if (!validation.valid) {
      return res.status(400).json({ error: validation.error });
    }

    const photoUrl = await uploadFile(
      req.file.buffer,
      req.file.originalname,
      `business-photos/${businessId}`
    );

    const business = businesses.update(businessId, { ownerPhotoUrl: photoUrl });
    if (!business) {
      return res.status(404).json({ error: 'Business not found' });
    }

    res.json({ success: true, photoUrl, business });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to upload photo' });
  }
});

// DELETE /api/businesses/:businessId/photo — Remove owner photo
router.delete('/', async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;
    const business = businesses.update(businessId, { ownerPhotoUrl: null });
    if (!business) {
      return res.status(404).json({ error: 'Business not found' });
    }
    res.json({ success: true, business });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete photo' });
  }
});

export default router;
