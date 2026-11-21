import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pixora Shield",
  description: "Blur faces and license plates in uploaded videos.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
