//! Random selection of samples: the training subset and the validation split.

use rand::Rng;
use rand::seq::SliceRandom;

/// Sample indices divided into a chosen part and the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// Chosen indices, ascending.
    pub chosen: Vec<usize>,
    /// Remaining indices, ascending.
    pub rest: Vec<usize>,
}

/// Chooses `count` of `indices` at random.
///
/// With `labels` (class index of every sample, indexed by sample index) the
/// choice is stratified: every class contributes the same share of its
/// samples, with quotas from [`stratified_quotas`]. Without labels all
/// indices are shuffled together. Both parts keep ascending order, so the
/// samples stay in file order.
pub fn split_indices<R: Rng + ?Sized>(
    indices: &[usize],
    count: usize,
    labels: Option<&[usize]>,
    rng: &mut R,
) -> Split {
    assert!(
        count <= indices.len(),
        "cannot choose {count} of {} samples",
        indices.len()
    );
    let mut chosen = Vec::with_capacity(count);
    let mut rest = Vec::with_capacity(indices.len() - count);
    match labels {
        None => take_random(indices.to_vec(), count, rng, &mut chosen, &mut rest),
        Some(labels) => {
            let groups = group_by_class(indices, labels);
            let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
            let quotas = stratified_quotas(&sizes, count);
            for (group, quota) in groups.into_iter().zip(quotas) {
                take_random(group, quota, rng, &mut chosen, &mut rest);
            }
        }
    }
    chosen.sort_unstable();
    rest.sort_unstable();
    Split { chosen, rest }
}

/// Splits `count` among classes of the given sizes proportionally, using the
/// largest remainder method: each class first gets `⌊size · count / n⌋`, the
/// leftover goes one by one to the classes with the largest remainders (ties
/// to the lower class index). The quotas sum to exactly `count`.
pub fn stratified_quotas(sizes: &[usize], count: usize) -> Vec<usize> {
    let total: usize = sizes.iter().sum();
    assert!(count <= total, "cannot choose {count} of {total} samples");
    if total == 0 {
        return vec![0; sizes.len()];
    }
    let mut quotas: Vec<usize> = sizes.iter().map(|size| size * count / total).collect();
    let leftover = count - quotas.iter().sum::<usize>();
    let mut by_remainder: Vec<usize> = (0..sizes.len()).collect();
    // Stable sort keeps lower class indices first among equal remainders.
    by_remainder.sort_by_key(|&class| std::cmp::Reverse(sizes[class] * count % total));
    for &class in &by_remainder[..leftover] {
        quotas[class] += 1;
    }
    quotas
}

/// Groups `indices` by class, in class order; classes without samples give
/// empty groups.
fn group_by_class(indices: &[usize], labels: &[usize]) -> Vec<Vec<usize>> {
    let class_count = indices.iter().map(|&i| labels[i] + 1).max().unwrap_or(0);
    let mut groups = vec![Vec::new(); class_count];
    for &index in indices {
        groups[labels[index]].push(index);
    }
    groups
}

/// Shuffles `indices`, moves the first `count` to `chosen` and the rest to
/// `rest`.
fn take_random<R: Rng + ?Sized>(
    mut indices: Vec<usize>,
    count: usize,
    rng: &mut R,
    chosen: &mut Vec<usize>,
    rest: &mut Vec<usize>,
) {
    indices.shuffle(rng);
    chosen.extend_from_slice(&indices[..count]);
    rest.extend_from_slice(&indices[count..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{RngStream, rng};

    /// 300 samples, 100 per class, classes interleaved: 0, 1, 2, 0, 1, 2, ...
    fn labels() -> Vec<usize> {
        (0..300).map(|i| i % 3).collect()
    }

    fn split(count: usize, labels: Option<&[usize]>, seed: u64) -> Split {
        let indices: Vec<usize> = (0..300).collect();
        split_indices(&indices, count, labels, &mut rng(seed, RngStream::Data))
    }

    fn assert_partition(split: &Split, indices: &[usize]) {
        let mut all: Vec<usize> = split.chosen.iter().chain(&split.rest).copied().collect();
        all.sort_unstable();
        assert_eq!(all, indices);
        assert!(split.chosen.is_sorted() && split.rest.is_sorted());
    }

    #[test]
    fn random_split_is_a_partition() {
        let split = split(60, None, 42);
        assert_eq!(split.chosen.len(), 60);
        assert_partition(&split, &(0..300).collect::<Vec<_>>());
    }

    #[test]
    fn same_seed_gives_same_split() {
        let labels = labels();
        assert_eq!(split(60, None, 42), split(60, None, 42));
        assert_eq!(split(60, Some(&labels), 7), split(60, Some(&labels), 7));
    }

    #[test]
    fn different_seeds_give_different_splits() {
        assert_ne!(split(60, None, 42), split(60, None, 43));
    }

    #[test]
    fn stratified_split_keeps_class_proportions() {
        let labels = labels();
        let split = split(60, Some(&labels), 42);
        assert_partition(&split, &(0..300).collect::<Vec<_>>());
        for class in 0..3 {
            let count = split.chosen.iter().filter(|&&i| labels[i] == class).count();
            assert_eq!(count, 20, "class {class}");
        }
    }

    #[test]
    fn stratified_split_of_a_subset_uses_global_labels() {
        let labels = labels();
        let indices: Vec<usize> = (0..300).step_by(2).collect();
        let split = split_indices(&indices, 30, Some(&labels), &mut rng(1, RngStream::Data));
        assert_partition(&split, &indices);
        for class in 0..3 {
            let count = split.chosen.iter().filter(|&&i| labels[i] == class).count();
            assert_eq!(count, 10, "class {class}");
        }
    }

    #[test]
    fn choosing_all_or_nothing() {
        let all = split(300, None, 42);
        assert_eq!((all.chosen.len(), all.rest.len()), (300, 0));
        let none = split(0, Some(&labels()), 42);
        assert_eq!((none.chosen.len(), none.rest.len()), (0, 300));
    }

    #[test]
    fn quotas_are_proportional() {
        assert_eq!(stratified_quotas(&[100, 100, 100], 60), [20, 20, 20]);
        assert_eq!(stratified_quotas(&[60, 30, 10], 10), [6, 3, 1]);
    }

    #[test]
    fn quotas_distribute_leftover_by_largest_remainder() {
        // Exact shares 2.5, 1.5, 1.0: the tie goes to the lower class index.
        assert_eq!(stratified_quotas(&[5, 3, 2], 5), [3, 1, 1]);
        // Exact shares 1.2, 0.6, 0.2 (sum 2): remainders 0.2, 0.6, 0.2.
        assert_eq!(stratified_quotas(&[6, 3, 1], 2), [1, 1, 0]);
    }

    #[test]
    fn quotas_sum_to_count() {
        let sizes = [7, 13, 1, 29, 0, 4];
        let total: usize = sizes.iter().sum();
        for count in 0..=total {
            assert_eq!(
                stratified_quotas(&sizes, count).iter().sum::<usize>(),
                count
            );
        }
    }

    #[test]
    #[should_panic(expected = "cannot choose 301 of 300 samples")]
    fn rejects_count_above_sample_count() {
        split(301, None, 42);
    }
}
