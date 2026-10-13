#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装甲板视频推理与关键点可视化脚本。

功能：
  1. 加载 YOLO26n-pose 装甲板权重（best.pt）；
  2. 读取自行拍摄的视频（也支持单张图片）；
  3. 逐帧推理，绘制检测框、类别、置信度，以及模型预测出的 4 个关键点；
  4. 保存与原始视频内容一一对应、可正常播放的结果视频；
  5. 输出统计报告（检测帧数、平均置信度、关键点平均置信度、耗时等）。

模型路径、视频路径、输出路径均可通过命令行参数修改。

示例：
  python infer.py --weights weights/best.pt --source videos/d414.mp4 --output runs/d414_result.mp4
  python infer.py --source videos/d414.mp4 --device cpu --imgsz 640 --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 关键点/检测框配色：每个目标使用同一种颜色，便于看出关键点属于哪个目标
PALETTE = [
    (0, 255, 0),      # 绿
    (255, 144, 30),   # 蓝
    (0, 215, 255),    # 黄
    (255, 0, 255),    # 品红
    (255, 255, 0),    # 青
    (60, 60, 255),    # 红
    (255, 0, 128),    # 紫
    (0, 128, 255),    # 橙
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="YOLO26n-pose 装甲板推理 + 检测框/类别/置信度/4 关键点可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- 路径参数（模型 / 视频 / 输出，均可修改）----
    parser.add_argument("--weights", type=Path, default=here / "weights" / "best.pt",
                        help="模型权重路径")
    parser.add_argument("--source", type=Path, default=here / "videos" / "d414.mp4",
                        help="输入视频（或图片）路径")
    parser.add_argument("--output", type=Path, default=None,
                        help="结果视频/图片保存路径，默认 runs/<源文件名>_result.<扩展名>")
    parser.add_argument("--report-json", type=Path, default=None,
                        help="可选：把本次运行的参数与统计结果写入 JSON")
    parser.add_argument("--save-txt", action="store_true",
                        help="可选：把每帧的框与关键点坐标写入 runs/<源文件名>_detections.txt")

    # ---- 推理参数 ----
    parser.add_argument("--device", default="cpu",
                        help="推理设备：cpu 或 0/0,1（GPU）")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=30, help="单帧最大检测数")
    parser.add_argument("--kpt-conf", type=float, default=0.0,
                        help="关键点置信度显示阈值，低于该值的点画成空心（默认全部实心显示）")

    # ---- 可视化参数 ----
    parser.add_argument("--line-width", type=int, default=2, help="检测框线宽")
    parser.add_argument("--kpt-radius", type=int, default=5, help="关键点半径")
    parser.add_argument("--font-scale", type=float, default=0.6, help="文字大小")
    parser.add_argument("--skeleton", choices=["hull", "order", "none"], default="hull",
                        help="关键点连线方式：hull=按 4 个预测点的凸包描出装甲板轮廓（默认，"
                             "不依赖关键点输出顺序）；order=按关键点顺序依次连线；none=只画点")
    parser.add_argument("--no-hud", action="store_true", help="不显示左上角信息面板")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="只处理前 N 帧（0 表示整段视频），用于快速自检")

    # ---- 其他 ----
    parser.add_argument("--codec", default="mp4v",
                        help="视频编码 fourcc，默认 mp4v（兼容性最好）")
    parser.add_argument("--show", action="store_true", help="实时弹窗预览（按 q 退出）")
    return parser.parse_args(argv)


def resolve_output(args) -> Path:
    """根据 --source 推导默认输出路径。"""
    here = Path(__file__).resolve().parent
    if args.output is not None:
        return args.output
    src = args.source
    if src.suffix.lower() in IMAGE_SUFFIXES:
        return here / "runs" / f"{src.stem}_result{src.suffix}"
    return here / "runs" / f"{src.stem}_result.mp4"


def instance_color(index: int):
    return PALETTE[index % len(PALETTE)]


def kpt_pairs(kpts_xy, kpts_conf):
    """把关键点整理成 [x, y, conf or None] 列表。

    本仓库使用的 yolo26n-pose 权重 kpt_shape=[4, 2]，即只有 (x, y)、没有关键点置信度，
    此时 kpts_conf 为 None，conf 记为 None（而不是丢掉关键点）。
    """
    if not kpts_xy:
        return []
    confs = list(kpts_conf) if kpts_conf else [None] * len(kpts_xy)
    return [[round(float(p[0]), 1), round(float(p[1]), 1),
             None if c is None else round(float(c), 3)]
            for p, c in zip(kpts_xy, confs)]


def draw_detection(frame, xyxy, cls_id, conf, kpts_xy, kpts_conf, names,
                   color, line_width, kpt_radius, font_scale,
                   skeleton="hull", kpt_conf_thr=0.0):
    """在帧上绘制一个目标的检测框、类别、置信度与 4 个关键点。

    kpts_xy / kpts_conf 直接来自模型输出，不使用检测框角点代替。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))

    # 1) 检测框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

    # 2) 类别 + 置信度标签（带底色，保证可见）
    label = f"{names.get(int(cls_id), str(cls_id))} {conf:.2f}"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_width - 1))
    ty = y1 - 6 if y1 - th - 6 > 0 else y2 + th + 8
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + base), color, -1)
    cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), max(1, line_width - 1), cv2.LINE_AA)

    # 3) 关键点（模型预测值）
    pts = []
    if kpts_xy is not None:
        for k in range(len(kpts_xy)):
            kx, ky = float(kpts_xy[k][0]), float(kpts_xy[k][1])
            if kx <= 0 and ky <= 0:      # 模型未给出该点
                pts.append(None)
                continue
            pts.append((int(round(kx)), int(round(ky))))

    # 3a) 先描出轮廓，明确这 4 个关键点属于同一个目标
    #     hull：按 4 个预测点的凸包描边（不依赖关键点的输出顺序，最稳）
    #     order：按关键点顺序依次连线（需要模型的关键点顺序本身就是沿周界的）
    valid = [p for p in pts if p is not None]
    if skeleton == "hull" and len(valid) >= 3:
        hull = cv2.convexHull(np.array(valid, dtype=np.int32))
        cv2.polylines(frame, [hull], True, color, max(1, line_width - 1), cv2.LINE_AA)
    elif skeleton == "order" and len(valid) >= 2:
        for a, b in zip(valid, valid[1:] + valid[:1]):
            cv2.line(frame, a, b, color, max(1, line_width - 1), cv2.LINE_AA)

    # 3b) 再画点：实心 = 置信度达标，空心 = 低于阈值
    for k, p in enumerate(pts):
        if p is None:
            continue
        kc = float(kpts_conf[k]) if kpts_conf is not None else 1.0
        if kc >= kpt_conf_thr:
            cv2.circle(frame, p, kpt_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p, kpt_radius, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, p, kpt_radius, color, 2, cv2.LINE_AA)
        cv2.putText(frame, str(k + 1), (p[0] + kpt_radius + 1, p[1] - kpt_radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(k + 1), (p[0] + kpt_radius + 1, p[1] - kpt_radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_hud(frame, lines):
    """左上角半透明信息面板。"""
    if not lines:
        return
    pad, lh = 8, 22
    scale, thick = 0.55, 1
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines) + pad * 2
    height = lh * len(lines) + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def open_writer(path: Path, fps: float, size, codec: str):
    """打开视频写入器，失败时回退到 XVID/AVI。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if writer.isOpened():
        return writer, path
    print(f"[warn] 编码器 {codec} 打开失败，回退 XVID(.avi)")
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    return writer, fallback


def build_report(args, model, elapsed, frames, det_frames, det_count,
                 conf_sum, kpt_conf_sum, kpt_count, kpt_xy_count, out_path):
    return {
        "weights": str(args.weights),
        "source": str(args.source),
        "output": str(out_path),
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "model_task": getattr(model, "task", None),
        "model_names": {int(k): v for k, v in dict(model.names).items()},
        "kpt_shape": list(getattr(model.model, "kpt_shape", []) or []),
        "frames_processed": frames,
        "frames_with_detection": det_frames,
        "detections_total": det_count,
        "mean_box_conf": round(conf_sum / det_count, 4) if det_count else None,
        "mean_kpt_conf": round(kpt_conf_sum / kpt_count, 4) if kpt_count else None,
        "keypoints_total": kpt_xy_count,
        "keypoints_per_target": round(kpt_xy_count / det_count, 2) if det_count else None,
        "elapsed_sec": round(elapsed, 3),
        "processing_fps": round(frames / elapsed, 2) if elapsed > 0 else None,
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.weights.is_file():
        print(f"[error] 找不到权重文件：{args.weights}")
        print("        请用 --weights 指定 best.pt 路径（见 README「模型权重准备」）。")
        return 2
    if not args.source.is_file():
        print(f"[error] 找不到输入文件：{args.source}")
        print("        请用 --source 指定自己拍摄的视频路径。")
        return 2

    out_path = resolve_output(args)
    is_image = args.source.suffix.lower() in IMAGE_SUFFIXES

    from ultralytics import YOLO

    print(f"[info] 加载权重：{args.weights}")
    model = YOLO(str(args.weights))
    names = dict(model.names)
    kpt_shape = list(getattr(model.model, "kpt_shape", []) or [])
    print(f"[info] 任务={getattr(model, 'task', '?')}  类别={names}  关键点形状={kpt_shape}")
    print(f"[info] 设备={args.device}  imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}")

    # ---------------- 单张图片 ----------------
    if is_image:
        frame = cv2.imread(str(args.source))
        if frame is None:
            print(f"[error] 无法读取图片：{args.source}")
            return 2
        res = model.predict(frame, device=args.device, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, max_det=args.max_det, verbose=False)[0]
        n = 0 if res.boxes is None else len(res.boxes)
        for i in range(n):
            xyxy = res.boxes.xyxy[i].tolist()
            clsid = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            kxy = res.keypoints.xy[i].tolist() if res.keypoints is not None else None
            kcf = res.keypoints.conf[i].tolist() if (res.keypoints is not None
                                                    and res.keypoints.conf is not None) else None
            draw_detection(frame, xyxy, clsid, conf, kxy, kcf, names,
                           instance_color(i), args.line_width, args.kpt_radius,
                           args.font_scale, args.skeleton, args.kpt_conf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        print(f"[info] 检测到 {n} 个目标，结果已保存：{out_path}")
        return 0

    # ---------------- 视频 ----------------
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"[error] 无法打开视频：{args.source}")
        print("        请确认文件未损坏、编码受 OpenCV 支持（推荐 MP4/H.264）。")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        print(f"[warn] 视频帧率异常({fps})，按 30 FPS 处理")
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] 输入视频：{args.source}  {width}x{height}  {fps:.2f} FPS  约 {total} 帧")

    writer, out_path = open_writer(out_path, fps, (width, height), args.codec)
    if not writer.isOpened():
        print("[error] 无法创建输出视频，请检查 --output 路径与 --codec 设置。")
        return 2
    print(f"[info] 结果视频将保存到：{out_path}")

    txt_lines = []
    frames = det_frames = det_count = kpt_count = kpt_xy_count = 0
    conf_sum = kpt_conf_sum = 0.0
    limit = args.max_frames if args.max_frames > 0 else (total if total > 0 else 10 ** 9)
    t0 = time.time()

    while frames < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        res = model.predict(frame, device=args.device, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, max_det=args.max_det, verbose=False)[0]
        n = 0 if res.boxes is None else len(res.boxes)
        if n > 0:
            det_frames += 1

        for i in range(n):
            xyxy = res.boxes.xyxy[i].tolist()
            clsid = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            kxy = res.keypoints.xy[i].tolist() if res.keypoints is not None else None
            kcf = res.keypoints.conf[i].tolist() if (res.keypoints is not None
                                                    and res.keypoints.conf is not None) else None
            det_count += 1
            conf_sum += conf
            kpt_xy_count += len(kxy or [])
            if kcf:
                kpt_count += len(kcf)
                kpt_conf_sum += sum(float(c) for c in kcf)
            draw_detection(frame, xyxy, clsid, conf, kxy, kcf, names,
                           instance_color(i), args.line_width, args.kpt_radius,
                           args.font_scale, args.skeleton, args.kpt_conf)
            if args.save_txt:
                txt_lines.append(
                    f"{frames} {names.get(clsid, clsid)} {conf:.4f} "
                    f"box={[round(v, 1) for v in xyxy]} "
                    f"kpts={kpt_pairs(kxy, kcf)}"
                )

        if not args.no_hud:
            draw_hud(frame, [
                f"frame {frames}   det {n}",
                f"{args.weights.name}  imgsz {args.imgsz}  conf {args.conf}",
                f"device {args.device}",
            ])

        writer.write(frame)
        if args.show:
            cv2.imshow("armor-inference (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[info] 用户中断预览")
                break
        if frames % 30 == 0:
            print(f"  ... {frames} 帧，累计检测 {det_count} 个", flush=True)

    elapsed = time.time() - t0
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    report = build_report(args, model, elapsed, frames, det_frames, det_count,
                          conf_sum, kpt_conf_sum, kpt_count, kpt_xy_count, out_path)
    if args.save_txt:
        txt_path = out_path.parent / f"{args.source.stem}_detections.txt"
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        print(f"[info] 检测明细已保存：{txt_path}")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] 统计报告已保存：{args.report_json}")

    print("\n===== 运行统计 =====")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if det_count == 0:
        # 全片无有效检测时的排查提示（并落盘首帧便于分析）
        debug = out_path.parent / f"{args.source.stem}_first_frame.jpg"
        cap2 = cv2.VideoCapture(str(args.source))
        ok, first = cap2.read()
        cap2.release()
        if ok:
            cv2.imwrite(str(debug), first)
            print(f"[warn] 全片未检出目标，已保存首帧用于排查：{debug}")
        print("[warn] 排查建议：1) 权重是否正确/是否为本任务的 yolo26n-pose best.pt；"
              "2) 视频是否可正常播放、内容是否为装甲板；3) 降低 --conf（如 0.10）；"
              "4) 调整 --imgsz（如 960）；5) 确认 --device 可用。")
    else:
        print(f"[info] 结果视频已保存：{out_path}（{frames} 帧，{det_frames} 帧检出目标）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
