import AppKit
import Darwin
import Foundation
import PDFKit
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 3 else {
    fail("Usage: ocr_pdf.swift <pdf-path> <comma-separated-page-numbers>")
}

let pdfURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard let document = PDFDocument(url: pdfURL) else {
    fail("PDFKit could not open the PDF.")
}

let pageNumbers = CommandLine.arguments[2]
    .split(separator: ",")
    .compactMap { Int($0) }
let pageSize = CGSize(width: 1700, height: 2200)
var result: [String: String] = [:]

for pageNumber in pageNumbers where pageNumber > 0 && pageNumber <= document.pageCount {
    guard let page = document.page(at: pageNumber - 1) else { continue }
    let image = page.thumbnail(of: pageSize, for: .cropBox)
    var proposedRect = CGRect(origin: .zero, size: image.size)
    guard let cgImage = image.cgImage(
        forProposedRect: &proposedRect,
        context: nil,
        hints: nil
    ) else {
        continue
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true

    do {
        try VNImageRequestHandler(cgImage: cgImage).perform([request])
        let observations = (request.results ?? []).sorted {
            if abs($0.boundingBox.maxY - $1.boundingBox.maxY) > 0.01 {
                return $0.boundingBox.maxY > $1.boundingBox.maxY
            }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }
        result[String(pageNumber)] = observations.compactMap {
            $0.topCandidates(1).first?.string
        }.joined(separator: "\n")
    } catch {
        FileHandle.standardError.write(
            Data("OCR failed on page \(pageNumber): \(error)\n".utf8)
        )
    }
}

do {
    let data = try JSONSerialization.data(withJSONObject: result)
    FileHandle.standardOutput.write(data)
} catch {
    fail("Could not encode OCR results.")
}
