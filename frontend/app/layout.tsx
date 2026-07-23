import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Mundpost — Google Review Service',
  description: 'Increase Google reviews for your business automatically',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
