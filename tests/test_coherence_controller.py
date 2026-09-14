import unittest

import torch

from models.coherence_controller import (
    CompactCoherenceRoPEController,
    CoherenceRoPEController,
    MaskAwareCoherenceDescriptor,
    RunningFeatureStandardizer,
)
from models.csi_mae import csi_mae_tiny
from util.video_vit import RoPEAttention3D


def patchify(csi, patch_size=4):
    batch, t, k, u = csi.shape
    p = patch_size
    return (
        csi.reshape(batch, t // p, p, k // p, p, u // p, p)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(batch, -1, p**3)
    )


class CoherenceDescriptorTest(unittest.TestCase):
    def setUp(self):
        self.builder = MaskAwareCoherenceDescriptor(include_metadata=True)
        self.input_size = torch.tensor([[8], [8], [8]])
        self.phys_meta = torch.tensor([[3.5e9, 30e3, 1e-3, 0.5 * 299792458 / 3.5e9]])

    def test_constant_channel_has_unit_correlation(self):
        csi = torch.ones(1, 8, 8, 8, dtype=torch.complex64)
        patches = patchify(csi)
        visible = torch.ones(1, patches.shape[1], dtype=torch.bool)
        descriptor = self.builder(patches, self.input_size, visible, self.phys_meta)
        torch.testing.assert_close(descriptor[0, :9], torch.ones(9), atol=1e-5, rtol=0)
        self.assertEqual(descriptor.shape[-1], self.builder.output_dim)

    def test_masked_patch_values_do_not_leak(self):
        generator = torch.Generator().manual_seed(7)
        real = torch.randn(1, 8, 8, 8, generator=generator)
        imag = torch.randn(1, 8, 8, 8, generator=generator)
        patches = patchify(torch.complex(real, imag))
        visible = torch.ones(1, patches.shape[1], dtype=torch.bool)
        visible[:, -1] = False
        modified = patches.clone()
        modified[:, -1] = 1e5 + 1e5j
        first = self.builder(patches, self.input_size, visible, self.phys_meta)
        second = self.builder(modified, self.input_size, visible, self.phys_meta)
        torch.testing.assert_close(first, second)

    def test_rho_is_invariant_to_common_phase_and_amplitude(self):
        compact = MaskAwareCoherenceDescriptor(
            include_metadata=False,
            feature_groups=("rho", "validity"),
            validity_mode="axis_mean",
        )
        csi = torch.randn(1, 8, 8, 8) + 1j * torch.randn(1, 8, 8, 8)
        patches = patchify(csi.to(torch.complex64))
        visible = torch.ones(1, patches.shape[1], dtype=torch.bool)
        reference = compact(patches, self.input_size, visible)
        transformed = patches * (3.0 * torch.exp(torch.tensor(0.7j)))
        changed = compact(transformed, self.input_size, visible)
        torch.testing.assert_close(reference[:, :9], changed[:, :9], atol=1e-5, rtol=1e-5)

    def test_validity_distinguishes_missing_pairs_from_low_correlation(self):
        compact = MaskAwareCoherenceDescriptor(
            include_metadata=False,
            feature_groups=("rho", "validity"),
            lags=(4,),
            validity_mode="axis_mean",
        )
        csi = torch.randn(1, 8, 8, 8, dtype=torch.complex64)
        patches = patchify(csi)
        none_visible = torch.zeros(1, patches.shape[1], dtype=torch.bool)
        missing = compact(patches, self.input_size, none_visible)
        self.assertTrue(torch.equal(missing[:, :3], torch.zeros_like(missing[:, :3])))
        self.assertTrue(torch.equal(missing[:, 3:], torch.zeros_like(missing[:, 3:])))

        visible = torch.ones_like(none_visible)
        observed = compact(patches, self.input_size, visible)
        self.assertTrue(torch.all(observed[:, 3:] > 0))

    def test_compact_layout_excludes_constant_spacing_and_raw_statistics(self):
        compact = MaskAwareCoherenceDescriptor(
            feature_groups=("rho", "validity", "physics_meta", "size_meta"),
            validity_mode="axis_mean",
        )
        self.assertNotIn("spacing_wavelength", compact.feature_names)
        self.assertNotIn("mean_real", compact.feature_names)
        self.assertNotIn("std_real", compact.feature_names)
        self.assertEqual(compact.output_dim, 18)


class CoherenceControllerTest(unittest.TestCase):
    def test_attention_capture_is_opt_in_and_output_preserving(self):
        attention = RoPEAttention3D(dim=16, num_heads=4).eval()
        inputs = torch.randn(2, 6, 16)
        with torch.no_grad():
            reference = attention(inputs)
            self.assertIsNone(attention.last_attn_map)
            attention.capture_attention = True
            captured_output = attention(inputs)
        torch.testing.assert_close(reference, captured_output)
        self.assertEqual(attention.last_attn_map.shape, (2, 4, 6, 6))

    def test_standardizer_calibration_is_frozen_and_clipped(self):
        standardizer = RunningFeatureStandardizer(2, clip_value=5.0)
        standardizer.begin_calibration()
        standardizer(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        standardizer.finalize_calibration()
        mean_before = standardizer.running_mean.clone()
        normalized = standardizer(torch.tensor([[1000.0, -1000.0]]))
        self.assertTrue(torch.equal(standardizer.running_mean, mean_before))
        self.assertLessEqual(float(normalized.abs().max()), 5.0)
        self.assertTrue(bool(standardizer.frozen.item()))
    def test_running_standardizer_tracks_between_batch_variation(self):
        standardizer = RunningFeatureStandardizer(1, momentum=0.5)
        standardizer.train()
        standardizer(torch.zeros(4, 1))
        standardizer(torch.full((4, 1), 2.0))
        self.assertGreater(float(standardizer.running_var), 0.1)

    def test_zero_initialized_controller_reduces_to_base_frequencies(self):
        controller = CoherenceRoPEController(30, num_heads=4, max_scale=4.0)
        controller.train()
        descriptor = torch.randn(3, 30)
        base = torch.randn(3, 4, 8)
        dynamic = controller(descriptor, base)
        torch.testing.assert_close(dynamic, base.unsqueeze(0).expand_as(dynamic))

    def test_compact_controller_supports_both_scale_granularities(self):
        descriptor = torch.randn(2, 12)
        tokens = torch.randn(2, 5, 32)
        base = torch.randn(3, 4, 4)
        for granularity in ("axis_head", "axis_head_frequency"):
            controller = CompactCoherenceRoPEController(
                descriptor_dim=12,
                token_dim=32,
                num_heads=4,
                head_dim=8,
                feature_groups=("rho",) * 9 + ("validity",) * 3,
                scale_granularity=granularity,
            )
            dynamic = controller(descriptor, tokens, base)
            torch.testing.assert_close(dynamic, base.unsqueeze(0).expand_as(dynamic))

    def test_compact_token_normalization_modes_use_distinct_inputs(self):
        tokens = torch.tensor(
            [
                [[1.0, 3.0, 2.0, 8.0], [3.0, 5.0, 6.0, 4.0]],
                [[2.0, 8.0, 1.0, 7.0], [6.0, 4.0, 5.0, 3.0]],
            ]
        )
        expected_mean = tokens.mean(dim=1)
        expected_std = torch.sqrt(tokens.var(dim=1, unbiased=False) + 1e-6)
        expected_raw = torch.cat((expected_mean, expected_std), dim=-1)

        raw = CompactCoherenceRoPEController(
            descriptor_dim=3,
            token_dim=4,
            num_heads=2,
            head_dim=2,
            feature_groups=("rho",) * 3,
            token_groups=("token_mean", "token_std"),
            token_normalization="raw",
        )
        torch.testing.assert_close(raw._token_context(tokens), expected_raw)

        frozen = CompactCoherenceRoPEController(
            descriptor_dim=3,
            token_dim=4,
            num_heads=2,
            head_dim=2,
            feature_groups=("rho",) * 3,
            token_groups=("token_mean", "token_std"),
            token_normalization="frozen",
        )
        frozen.token_standardizer.running_mean.fill_(1.0)
        frozen.token_standardizer.running_var.fill_(4.0)
        frozen.token_standardizer.frozen.fill_(True)
        torch.testing.assert_close(
            frozen._token_context(tokens), (expected_raw - 1.0) / 2.0
        )

        sample = CompactCoherenceRoPEController(
            descriptor_dim=3,
            token_dim=4,
            num_heads=2,
            head_dim=2,
            feature_groups=("rho",) * 3,
            token_groups=("token_mean", "token_std"),
            token_normalization="sample",
        )
        context = sample._token_context(tokens)
        for group in context.split(4, dim=-1):
            torch.testing.assert_close(
                group.mean(dim=-1), torch.zeros(2), atol=1e-5, rtol=0
            )

    def test_raw_validity_bypasses_descriptor_standardization(self):
        controller = CompactCoherenceRoPEController(
            descriptor_dim=6,
            token_dim=4,
            num_heads=2,
            head_dim=2,
            feature_groups=("rho",) * 3 + ("validity",) * 3,
            raw_descriptor_groups=("validity",),
            hidden_dim=6,
            architecture="direct_linear",
        )
        controller.standardizer.running_mean.copy_(
            torch.tensor([0.5, 0.5, 0.5, 0.2, 0.4, 0.6])
        )
        controller.standardizer.running_var.fill_(0.25)
        controller.standardizer.frozen.fill_(True)
        captured = {}

        def capture(_module, inputs):
            captured["context"] = inputs[0].detach().clone()

        hook = controller.output.register_forward_pre_hook(capture)
        descriptor = torch.tensor([[1.0, 0.0, 0.5, 0.0, 0.25, 1.0]])
        controller(
            descriptor,
            torch.zeros(1, 2, 4),
            torch.ones(3, 2, 1),
        )
        hook.remove()
        torch.testing.assert_close(
            captured["context"][:, :3],
            torch.tensor([[1.0, -1.0, 0.0]]),
        )
        torch.testing.assert_close(
            captured["context"][:, 3:], descriptor[:, 3:]
        )

    def test_constant_context_is_sample_independent_and_parameter_matched(self):
        common = dict(
            descriptor_dim=6,
            token_dim=8,
            num_heads=2,
            head_dim=4,
            feature_groups=("rho",) * 3 + ("validity",) * 3,
            token_groups=("token_mean", "token_std"),
            hidden_dim=8,
            token_hidden_dim=8,
        )
        sample = CompactCoherenceRoPEController(context_mode="sample", **common)
        constant = CompactCoherenceRoPEController(context_mode="constant", **common)
        self.assertEqual(
            sum(parameter.numel() for parameter in sample.parameters()),
            sum(parameter.numel() for parameter in constant.parameters()),
        )
        torch.manual_seed(4)
        for parameter in constant.parameters():
            if parameter.requires_grad:
                parameter.data.normal_(0.0, 0.1)
        base = torch.ones(3, 2, 2)
        first = constant(
            torch.randn(2, 6), torch.randn(2, 5, 8), base
        )
        second = constant(
            torch.randn(2, 6) * 100, torch.randn(2, 5, 8) * 100, base
        )
        torch.testing.assert_close(first, second)

    def test_token_statistics_ignore_invalid_padding(self):
        controller = CompactCoherenceRoPEController(
            descriptor_dim=3,
            token_dim=4,
            num_heads=2,
            head_dim=2,
            feature_groups=("rho",) * 3,
            token_groups=("token_mean", "token_std"),
            token_normalization="raw",
        )
        valid = torch.randn(2, 3, 4)
        padded = torch.cat((valid, torch.full((2, 2, 4), 1e6)), dim=1)
        reference = controller._token_context(valid)
        masked = controller._token_context(
            padded,
            token_mask=torch.tensor(
                [[True, True, True, False, False]] * 2, dtype=torch.bool
            ),
        )
        torch.testing.assert_close(reference, masked)

    def test_head_pool_reduces_token_context_to_two_values_per_head(self):
        controller = CompactCoherenceRoPEController(
            descriptor_dim=3,
            token_dim=12,
            num_heads=3,
            head_dim=4,
            feature_groups=("rho",) * 3,
            token_groups=("token_mean", "token_std"),
            token_pool="head",
            token_hidden_dim=64,
        )
        context = controller._token_context(torch.randn(2, 5, 12))
        self.assertEqual(context.shape, (2, 6))
        self.assertEqual(controller.token_projection[0].in_features, 6)

    def test_compact_controller_architectures_are_zero_initialized_and_trainable(self):
        descriptor = torch.randn(2, 15)
        tokens = torch.randn(2, 5, 48)
        base = torch.randn(3, 6, 4)
        for architecture in ("dual_branch", "fused_mlp", "direct_linear"):
            controller = CompactCoherenceRoPEController(
                descriptor_dim=15,
                token_dim=48,
                num_heads=6,
                head_dim=8,
                feature_groups=("rho",) * 9
                + ("validity",) * 3
                + ("size_meta",) * 3,
                token_groups=("token_mean", "token_std"),
                token_pool="head",
                hidden_dim=64,
                token_hidden_dim=64,
                architecture=architecture,
                fusion_hidden_dim=64,
            )
            dynamic = controller(descriptor, tokens, base)
            torch.testing.assert_close(dynamic, base.unsqueeze(0).expand_as(dynamic))
            dynamic.square().mean().backward()
            self.assertIsNotNone(controller.output.weight.grad)

    def test_s1_structural_variants_have_expected_parameter_counts(self):
        def count(token_dim, heads, architecture, token_pool, hidden, token_hidden):
            controller = CompactCoherenceRoPEController(
                descriptor_dim=15,
                token_dim=token_dim,
                num_heads=heads,
                head_dim=token_dim // heads,
                feature_groups=("rho",) * 9
                + ("validity",) * 3
                + ("size_meta",) * 3,
                token_groups=("token_mean", "token_std"),
                token_pool=token_pool,
                hidden_dim=hidden,
                token_hidden_dim=token_hidden,
                architecture=architecture,
                fusion_hidden_dim=64,
            )
            return sum(parameter.numel() for parameter in controller.parameters())

        pairs = ((768, 12), (512, 16))
        expected = {
            "s1": 454292,
            "lowrank": 176852,
            "head": 16596,
            "fused": 11092,
            "linear": 3744,
        }
        measured = {
            "s1": sum(
                count(dim, heads, "dual_branch", "channel", 128, 0)
                for dim, heads in pairs
            ),
            "lowrank": sum(
                count(dim, heads, "dual_branch", "channel", 64, 64)
                for dim, heads in pairs
            ),
            "head": sum(
                count(dim, heads, "dual_branch", "head", 64, 64)
                for dim, heads in pairs
            ),
            "fused": sum(
                count(dim, heads, "fused_mlp", "head", 64, 64)
                for dim, heads in pairs
            ),
            "linear": sum(
                count(dim, heads, "direct_linear", "head", 64, 64)
                for dim, heads in pairs
            ),
        }
        self.assertEqual(measured, expected)

    def test_scale_range_is_reciprocal_and_zero_logits_give_one(self):
        controller = CompactCoherenceRoPEController(
            descriptor_dim=3,
            token_dim=4,
            num_heads=2,
            head_dim=2,
            feature_groups=("rho",) * 3,
            max_scale=5.0,
        )
        descriptor = torch.randn(2, 3)
        base = torch.ones(3, 2, 1)
        controller(descriptor, torch.randn(2, 4, 4), base)
        torch.testing.assert_close(controller.last_scale, torch.ones_like(controller.last_scale))
        controller.output.bias.data.fill_(100.0)
        controller(descriptor, torch.randn(2, 4, 4), base)
        self.assertLessEqual(float(controller.last_scale.max()), 5.0)
        self.assertGreater(float(controller.last_scale.min()), 4.99)
        controller.output.bias.data.fill_(-100.0)
        controller(descriptor, torch.randn(2, 4, 4), base)
        self.assertGreaterEqual(float(controller.last_scale.min()), 0.19999)
        self.assertLess(float(controller.last_scale.max()), 0.201)

    def test_group_neutralization_uses_calibrated_mean(self):
        controller = CoherenceRoPEController(
            2,
            num_heads=1,
            feature_groups=("rho", "power"),
            neutralize_groups=("power",),
        )
        controller.standardizer.running_mean.copy_(torch.tensor([2.0, 7.0]))
        controller(torch.tensor([[1.0, 100.0]]), torch.ones(3, 1, 2))
        self.assertEqual(float(controller.last_descriptor[0, 1]), 7.0)

    def test_model_forward_and_backward(self):
        torch.manual_seed(3)
        model = csi_mae_tiny(
            rope_mode="adaptive",
            controller_mode="coherence_meta",
            pos_emb_type="ComplexRotation",
            decoder_pos_emb_type="SinCos_3D",
        )
        csi = torch.randn(2, 8, 8, 8) + 1j * torch.randn(2, 8, 8, 8)
        patches = patchify(csi.to(torch.complex64))
        token_length = torch.full((2,), patches.shape[1], dtype=torch.long)
        input_size = torch.tensor([[8, 8], [8, 8], [8, 8]])
        phys_meta = torch.tensor(
            [
                [3.5e9, 30e3, 1e-3, 0.5 * 299792458 / 3.5e9],
                [2.1e9, 15e3, 0.5e-3, 0.5 * 299792458 / 2.1e9],
            ]
        )
        loss, prediction, mask = model(
            patches,
            token_length,
            input_size=input_size,
            mask_ratio=0.5,
            mask_strategy="random",
            phys_meta=phys_meta,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(prediction.shape[:2], patches.shape[:2])
        self.assertEqual(mask.shape, patches.shape[:2])
        loss.backward()
        self.assertIsNotNone(model.enc_rope_controller.output.weight.grad)

    def test_compact_model_forward_and_backward(self):
        model = csi_mae_tiny(
            rope_mode="adaptive",
            controller_mode="compact",
            controller_feature_groups="rho,validity,physics_meta,size_meta",
            controller_lags=(1, 2, 4),
            controller_validity_mode="axis_mean",
            controller_scale_granularity="axis_head",
            pos_emb_type="ComplexRotation",
            decoder_pos_emb_type="SinCos_3D",
        )
        csi = torch.randn(2, 8, 8, 8) + 1j * torch.randn(2, 8, 8, 8)
        patches = patchify(csi.to(torch.complex64))
        loss, _, _ = model(
            patches,
            torch.full((2,), patches.shape[1], dtype=torch.long),
            input_size=torch.tensor([[8, 8], [8, 8], [8, 8]]),
            mask_ratio=0.5,
            mask_strategy="random",
            phys_meta=torch.tensor(
                [
                    [3.5e9, 30e3, 1e-3, 0.05],
                    [2.1e9, 15e3, 0.5e-3, 0.07],
                ]
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.enc_rope_controller.output.weight.grad)

    def test_adaptive_scope_disables_only_the_requested_controller(self):
        common = dict(
            rope_mode="adaptive",
            controller_mode="compact",
            controller_feature_groups="rho,validity",
            controller_token_groups="token_mean,token_std",
            pos_emb_type="ComplexRotation",
            decoder_pos_emb_type="SinCos_3D",
        )
        encoder_only = csi_mae_tiny(adaptive_scope="encoder_only", **common)
        self.assertIsNotNone(encoder_only.enc_rope_controller)
        self.assertIsNone(encoder_only.dec_rope_controller)
        decoder_only = csi_mae_tiny(adaptive_scope="decoder_only", **common)
        self.assertIsNone(decoder_only.enc_rope_controller)
        self.assertIsNotNone(decoder_only.dec_rope_controller)

    def test_masked_targets_do_not_change_compact_token_statistics(self):
        torch.manual_seed(11)
        model = csi_mae_tiny(
            rope_mode="adaptive",
            controller_mode="compact",
            controller_feature_groups="rho,validity,size_meta",
            controller_token_groups="token_mean,token_std",
            controller_token_normalization="frozen",
            controller_max_scale=5.0,
            pos_emb_type="ComplexRotation",
            decoder_pos_emb_type="SinCos_3D",
        ).eval()
        csi = torch.randn(1, 8, 8, 8) + 1j * torch.randn(1, 8, 8, 8)
        patches = patchify(csi.to(torch.complex64))
        lengths = torch.tensor([patches.shape[1]])
        dims = torch.tensor([[8], [8], [8]])
        meta = torch.tensor([[3.5e9, 30e3, 1e-3, 0.05]])
        with torch.no_grad():
            _, _, mask = model(
                patches,
                lengths,
                input_size=dims,
                mask_ratio=0.5,
                mask_strategy="temporal",
                phys_meta=meta,
            )
            reference_descriptor = model.enc_rope_controller.last_descriptor.clone()
            reference_encoder = model.enc_rope_controller.last_token_raw.clone()
            reference_decoder = model.dec_rope_controller.last_token_raw.clone()
            modified = patches.clone()
            modified[mask.bool()] = 1e5 + 1e5j
            model(
                modified,
                lengths,
                input_size=dims,
                mask_ratio=0.5,
                mask_strategy="temporal",
                phys_meta=meta,
            )
        torch.testing.assert_close(
            model.enc_rope_controller.last_descriptor, reference_descriptor
        )
        torch.testing.assert_close(
            model.enc_rope_controller.last_token_raw, reference_encoder
        )
        torch.testing.assert_close(
            model.dec_rope_controller.last_token_raw, reference_decoder
        )

    def test_visible_only_decoder_statistics_exclude_mask_placeholders(self):
        torch.manual_seed(19)
        model = csi_mae_tiny(
            rope_mode="adaptive",
            controller_mode="compact",
            controller_feature_groups="rho",
            controller_lags=(1,),
            controller_validity_mode="none",
            controller_token_groups="token_mean,token_std",
            controller_token_normalization="frozen",
            controller_decoder_token_scope="visible_only",
            pos_emb_type="None",
            decoder_pos_emb_type="None",
        ).eval()
        csi = torch.randn(1, 8, 8, 8) + 1j * torch.randn(1, 8, 8, 8)
        patches = patchify(csi.to(torch.complex64))
        lengths = torch.tensor([patches.shape[1]])
        dims = torch.tensor([[8], [8], [8]])

        with torch.no_grad():
            torch.manual_seed(23)
            model(
                patches,
                lengths,
                input_size=dims,
                mask_ratio=0.5,
                mask_strategy="temporal",
            )
            reference = model.dec_rope_controller.last_token_raw.clone()
            model.mask_token.fill_(1000.0)
            torch.manual_seed(23)
            model(
                patches,
                lengths,
                input_size=dims,
                mask_ratio=0.5,
                mask_strategy="temporal",
            )
        torch.testing.assert_close(
            model.dec_rope_controller.last_token_raw, reference
        )

    def test_all_valid_decoder_statistics_include_mask_placeholders(self):
        torch.manual_seed(29)
        model = csi_mae_tiny(
            rope_mode="adaptive",
            controller_mode="compact",
            controller_feature_groups="rho",
            controller_lags=(1,),
            controller_validity_mode="none",
            controller_token_groups="token_mean,token_std",
            controller_token_normalization="frozen",
            controller_decoder_token_scope="all_valid",
            pos_emb_type="None",
            decoder_pos_emb_type="None",
        ).eval()
        csi = torch.randn(1, 8, 8, 8) + 1j * torch.randn(1, 8, 8, 8)
        patches = patchify(csi.to(torch.complex64))
        lengths = torch.tensor([patches.shape[1]])
        dims = torch.tensor([[8], [8], [8]])

        with torch.no_grad():
            torch.manual_seed(31)
            model(
                patches,
                lengths,
                input_size=dims,
                mask_ratio=0.5,
                mask_strategy="temporal",
            )
            reference = model.dec_rope_controller.last_token_raw.clone()
            model.mask_token.fill_(1000.0)
            torch.manual_seed(31)
            model(
                patches,
                lengths,
                input_size=dims,
                mask_ratio=0.5,
                mask_strategy="temporal",
            )
        self.assertFalse(
            torch.allclose(model.dec_rope_controller.last_token_raw, reference)
        )

    def test_structured_masking_keeps_padding_indices_in_range(self):
        model = csi_mae_tiny(
            rope_mode="adaptive",
            controller_mode="compact",
            controller_feature_groups="rho,validity",
            controller_lags=(1, 2, 4),
            controller_validity_mode="axis_mean",
            controller_scale_granularity="axis_head",
            pos_emb_type="ComplexRotation",
            decoder_pos_emb_type="SinCos_3D",
        )
        max_length = 8
        tokens = torch.randn(2, max_length, model.embed_dim)
        # The second sample is shorter, which forces padded keep slots when a
        # structured mask is batched with the first sample.
        input_size = torch.tensor([[8, 4], [8, 8], [8, 8]])
        for masking in (model.temporal_masking, model.freq_masking):
            _, _, _, ids_keep, is_valid = masking(tokens, input_size, 0.5)
            self.assertTrue(torch.all(ids_keep >= 0))
            self.assertTrue(torch.all(ids_keep < max_length))
            self.assertTrue(torch.any(~is_valid))

    def test_axial_backbone_is_parameter_matched_and_trainable(self):
        common = dict(
            rope_mode="learnable",
            pos_emb_type="ComplexRotation",
            decoder_pos_emb_type="SinCos_3D",
        )
        global_model = csi_mae_tiny(attention_backbone="global", **common)
        axial_model = csi_mae_tiny(attention_backbone="axial", **common)
        global_parameters = sum(parameter.numel() for parameter in global_model.parameters())
        axial_parameters = sum(parameter.numel() for parameter in axial_model.parameters())
        self.assertEqual(global_parameters, axial_parameters)

        csi = torch.randn(1, 8, 8, 8) + 1j * torch.randn(1, 8, 8, 8)
        patches = patchify(csi.to(torch.complex64))
        loss, _, _ = axial_model(
            patches,
            torch.tensor([patches.shape[1]]),
            input_size=torch.tensor([[8], [8], [8]]),
            mask_ratio=0.5,
            mask_strategy="random",
        )
        self.assertTrue(torch.isfinite(loss))

    def test_fixed_1d_rope_uses_legacy_checkpoint_shape(self):
        model = csi_mae_tiny(
            rope_mode="fixed",
            rope_axes="1d",
            rope_theta=10000,
            pos_emb_type="ComplexRotation",
            decoder_pos_emb_type="SinCos_3D",
        )
        self.assertEqual(model.freqs.ndim, 2)
        self.assertEqual(model.freqs.shape[0], model.num_heads)



if __name__ == "__main__":
    unittest.main()
