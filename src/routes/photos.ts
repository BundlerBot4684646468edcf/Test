import express from 'express';
import multer from 'multer';
import prisma from '../db';
import { uploadFile, validatePhotoFile } from '../services/fileStorage';

const router = express.Router({ mergeParams: true });
const upload = multer({ storage: multer.memoryStorage() });

// POST /api/businesses/:id/photo — Upload owner photo
router.post('/', upload.single('photo'), async (req, res) => {
  try {
    const { id } = req.params;

    if (!req.file) {
      return res.status(400).json({ error: 'No file provided' });
    }

    // Validate file
    const validation = await validatePhotoFile({
      mimetype: req.file.mimetype,
      size: req.file.size,
    });

    if (!validation.valid) {
      return res.status(400).json({ error: validation.error });
    }

    // Upload file
    const photoUrl = await uploadFile(
      req.file.buffer,
      req.file.originalname,
      `business-photos/${id}`
    );

    // Save to database
    const business = await prisma.business.update({
      where: { id },
      data: { ownerPhotoUrl: photoUrl },
    });

    res.json({ success: true, photoUrl, business });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to upload photo' });
  }
});

// DELETE /api/businesses/:id/photo — Remove owner photo
router.delete('/', async (req, res) => {
  try {
    const { id } = req.params;

    const business = await prisma.business.update({
      where: { id },
      data: { ownerPhotoUrl: null },
    });

    res.json({ success: true, business });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete photo' });
  }
});

export default router;
