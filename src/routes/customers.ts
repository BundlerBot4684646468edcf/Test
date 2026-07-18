import express from 'express';
import multer from 'multer';
import prisma from '../db';
import { importCustomersFromCSV } from '../services/csvImport';

const router = express.Router({ mergeParams: true });
const upload = multer({ storage: multer.memoryStorage() });

// GET /api/businesses/:businessId/customers — List customers
router.get('/', async (req, res) => {
  try {
    const { businessId } = req.params;
    const { skip = 0, take = 20 } = req.query;

    const customers = await prisma.customer.findMany({
      where: { businessId },
      skip: parseInt(skip as string),
      take: parseInt(take as string),
      include: {
        reviewRequests: true,
      },
    });

    const total = await prisma.customer.count({ where: { businessId } });

    res.json({ customers, total });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to list customers' });
  }
});

// POST /api/businesses/:businessId/customers/import — Import from CSV
router.post('/import', upload.single('file'), async (req, res) => {
  try {
    const { businessId } = req.params;
    const { source = 'past' } = req.body;

    if (!req.file) {
      return res.status(400).json({ error: 'No file provided' });
    }

    const csvContent = req.file.buffer.toString('utf-8');
    const result = await importCustomersFromCSV(businessId, csvContent, source);

    res.json({
      success: true,
      imported: result.imported,
      errors: result.errors,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: String(error) });
  }
});

// DELETE /api/businesses/:businessId/customers/:customerId
router.delete('/:customerId', async (req, res) => {
  try {
    const { businessId, customerId } = req.params;

    const customer = await prisma.customer.findUnique({
      where: { id: customerId },
    });

    if (!customer || customer.businessId !== businessId) {
      return res.status(404).json({ error: 'Customer not found' });
    }

    await prisma.customer.delete({
      where: { id: customerId },
    });

    res.json({ success: true });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete customer' });
  }
});

export default router;
