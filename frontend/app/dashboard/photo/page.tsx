'use client';

import { useEffect, useRef, useState } from 'react';
import api from '@/lib/api';

const card: React.CSSProperties = {
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 14,
};

// Sign box in relative 0..1 coordinates over the photo.
interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
}

const DEFAULT_BOX: Box = { x: 0.4, y: 0.32, width: 0.18, height: 0.13, rotation: 0 };

export default function PhotoPage() {
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [box, setBox] = useState<Box>(DEFAULT_BOX);
  const [name, setName] = useState('Anna');
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ mode: 'move' | 'resize'; startX: number; startY: number; box: Box } | null>(null);

  // Load existing photo + sign box
  useEffect(() => {
    const id = localStorage.getItem('businessId');
    if (!id) return;
    api.get(`/businesses/${id}`).then((r) => {
      const b = r.data;
      if (b.ownerPhotoUrl) setPhotoUrl(b.ownerPhotoUrl);
      if (b.signX != null) {
        setBox({
          x: b.signX,
          y: b.signY ?? 0.32,
          width: b.signWidth ?? 0.18,
          height: b.signHeight ?? 0.13,
          rotation: b.signRotation ?? 0,
        });
      }
    }).catch(console.error);
  }, []);

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    setMsg(null);
    try {
      const id = localStorage.getItem('businessId');
      const fd = new FormData();
      fd.append('photo', file);
      const r = await api.post(`/businesses/${id}/photo`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPhotoUrl(r.data.photoUrl);
      setFile(null);
      setMsg({ ok: true, text: 'Foto hochgeladen. Jetzt das Schild markieren.' });
    } catch {
      setMsg({ ok: false, text: 'Upload fehlgeschlagen.' });
    } finally {
      setUploading(false);
    }
  };

  // Pointer drag/resize of the sign box
  const onPointerDown = (mode: 'move' | 'resize') => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { mode, startX: e.clientX, startY: e.clientY, box: { ...box } };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current || !wrapRef.current) return;
    const rect = wrapRef.current.getBoundingClientRect();
    const dx = (e.clientX - drag.current.startX) / rect.width;
    const dy = (e.clientY - drag.current.startY) / rect.height;
    const b = drag.current.box;
    if (drag.current.mode === 'move') {
      setBox({
        ...b,
        x: Math.min(Math.max(b.x + dx, 0), 1 - b.width),
        y: Math.min(Math.max(b.y + dy, 0), 1 - b.height),
      });
    } else {
      setBox({
        ...b,
        width: Math.min(Math.max(b.width + dx, 0.05), 1 - b.x),
        height: Math.min(Math.max(b.height + dy, 0.04), 1 - b.y),
      });
    }
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const saveBox = async () => {
    setSaving(true);
    setMsg(null);
    try {
      const id = localStorage.getItem('businessId');
      await api.put(`/businesses/${id}/photo/sign-box`, box);
      setMsg({ ok: true, text: 'Schild-Position gespeichert ✓' });
      showPreview();
    } catch {
      setMsg({ ok: false, text: 'Speichern fehlgeschlagen.' });
    } finally {
      setSaving(false);
    }
  };

  const showPreview = () => {
    const id = localStorage.getItem('businessId');
    const base = api.defaults.baseURL || '';
    setPreviewSrc(
      `${base}/businesses/${id}/photo/personalized?name=${encodeURIComponent(name)}&t=${Date.now()}`
    );
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight" style={{ color: 'var(--ink)' }}>
        Foto & Schild-Personalisierung
      </h1>

      {/* Upload */}
      <div style={card} className="p-6">
        <h2 className="text-sm font-semibold" style={{ color: 'var(--ink)' }}>
          1. Foto hochladen
        </h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>
          Inhaber hält ein leeres weißes Schild hoch. Der Name des Kunden wird später daraufgeschrieben.
        </p>
        <div className="mt-4 rounded-xl p-6 text-center" style={{ border: '1.5px dashed var(--line)', background: 'var(--plane)' }}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full text-sm"
            style={{ color: 'var(--ink-2)' }}
          />
        </div>
        {file && (
          <button
            onClick={upload}
            disabled={uploading}
            className="mt-4 w-full py-2.5 rounded-lg font-medium text-white disabled:opacity-60"
            style={{ background: 'var(--brand)' }}
          >
            {uploading ? 'Lädt…' : 'Hochladen'}
          </button>
        )}
      </div>

      {/* Calibrate */}
      {photoUrl && (
        <div style={card} className="p-6">
          <h2 className="text-sm font-semibold" style={{ color: 'var(--ink)' }}>
            2. Schild markieren
          </h2>
          <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>
            Zieh das rote Rechteck genau auf die weiße Fläche des Schilds. Ecke unten rechts = Größe ändern.
          </p>

          <div
            ref={wrapRef}
            className="mt-4 relative select-none"
            style={{ maxWidth: 480, touchAction: 'none' }}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={photoUrl} alt="Foto" style={{ width: '100%', borderRadius: 10, display: 'block' }} draggable={false} />
            <div
              onPointerDown={onPointerDown('move')}
              style={{
                position: 'absolute',
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.width * 100}%`,
                height: `${box.height * 100}%`,
                border: '2px solid #d63031',
                background: 'rgba(214,48,49,0.15)',
                cursor: 'move',
                transform: `rotate(${box.rotation}deg)`,
                transformOrigin: 'center',
              }}
            >
              <div
                onPointerDown={onPointerDown('resize')}
                style={{
                  position: 'absolute',
                  right: -8,
                  bottom: -8,
                  width: 16,
                  height: 16,
                  borderRadius: 8,
                  background: '#d63031',
                  cursor: 'nwse-resize',
                }}
              />
            </div>
          </div>

          {/* Rotation */}
          <div className="mt-4 flex items-center gap-3">
            <label className="text-sm" style={{ color: 'var(--ink-2)' }}>Drehung: {box.rotation}°</label>
            <input
              type="range"
              min={-15}
              max={15}
              value={box.rotation}
              onChange={(e) => setBox({ ...box, rotation: Number(e.target.value) })}
              className="flex-1"
            />
          </div>

          <button
            onClick={saveBox}
            disabled={saving}
            className="mt-4 w-full py-2.5 rounded-lg font-medium text-white disabled:opacity-60"
            style={{ background: 'var(--brand)' }}
          >
            {saving ? 'Speichert…' : 'Position speichern & Vorschau'}
          </button>
        </div>
      )}

      {/* Preview */}
      {photoUrl && (
        <div style={card} className="p-6">
          <h2 className="text-sm font-semibold" style={{ color: 'var(--ink)' }}>
            3. Vorschau mit Namen
          </h2>
          <div className="mt-3 flex items-center gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Name z.B. Anna"
              className="flex-1 px-3 py-2 rounded-lg text-sm"
              style={{ border: '1px solid var(--border)', background: 'var(--plane)', color: 'var(--ink)' }}
            />
            <button
              onClick={showPreview}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ border: '1px solid var(--border)', color: 'var(--ink-2)' }}
            >
              Vorschau
            </button>
          </div>
          {previewSrc && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewSrc} alt="Vorschau" style={{ width: '100%', maxWidth: 480, marginTop: 16, borderRadius: 10 }} />
          )}
        </div>
      )}

      {msg && (
        <div
          className="p-3 rounded-lg text-sm"
          style={{
            color: msg.ok ? 'var(--good-ink)' : 'var(--critical)',
            background: msg.ok ? 'rgba(12,163,12,0.10)' : 'rgba(208,59,59,0.10)',
          }}
        >
          {msg.text}
        </div>
      )}
    </div>
  );
}
