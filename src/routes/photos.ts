import express from 'express';
import multer from 'multer';
import { promises as fsp } from 'fs';
import { businesses } from '../db';
import {
  uploadFile,
  validatePhotoFile,
  localPathForUrl,
} from '../services/fileStorage';
import {
  personalizePhoto,
  DEFAULT_SIGN_BOX,
} from '../services/photoPersonalization';

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

// PUT /api/businesses/:businessId/photo/sign-box — Where the blank sign sits
// in the photo (relative 0..1 coords + rotation in degrees).
router.put('/sign-box', async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;
    const { x, y, width, height, rotation = 0 } = req.body;

    const nums = [x, y, width, height, rotation].map(Number);
    if (nums.some((n) => Number.isNaN(n))) {
      return res
        .status(400)
        .json({ error: 'x, y, width, height (0..1) and rotation are required numbers' });
    }

    const business = businesses.update(businessId, {
      signX: nums[0],
      signY: nums[1],
      signWidth: nums[2],
      signHeight: nums[3],
      signRotation: nums[4],
    });
    if (!business) return res.status(404).json({ error: 'Business not found' });

    res.json({ success: true, business });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to save sign box' });
  }
});

// GET /api/businesses/:businessId/photo/personalized?name=Anna
// Renders the owner photo with the customer's name written on the sign.
router.get('/personalized', async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;
    const name = ((req.query.name as string) || 'Anna').slice(0, 40);

    const business = businesses.get(businessId);
    if (!business) return res.status(404).json({ error: 'Business not found' });
    if (!business.ownerPhotoUrl) {
      return res
        .status(400)
        .json({ error: 'Kein Foto hochgeladen. Erst POST /photo verwenden.' });
    }

    const filePath = localPathForUrl(business.ownerPhotoUrl);
    if (!filePath) {
      return res.status(400).json({
        error: 'Foto liegt nicht im lokalen Speicher (R2-Fotos: Personalisierung folgt beim Versand).',
      });
    }

    const baseImage = await fsp.readFile(filePath);
    const box =
      business.signX != null
        ? {
            x: business.signX,
            y: business.signY ?? 0.6,
            width: business.signWidth ?? 0.5,
            height: business.signHeight ?? 0.2,
            rotation: business.signRotation ?? 0,
          }
        : DEFAULT_SIGN_BOX;

    const png = await personalizePhoto(baseImage, name, box);
    res.type('png').send(png);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to personalize photo' });
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
